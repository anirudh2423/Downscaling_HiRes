"""EDM diffusion training loop for ERA5-conditioned AORC temperature."""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time

from config import CFG
from dataset import AORCDataset
from model import build_model
from diffusion import EDM
from gpu_monitor import GPUMonitor


def train():
    torch.manual_seed(CFG.seed)
    device  = torch.device(CFG.device)
    out_dir = Path(CFG.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True

    # ── data ──────────────────────────────────────────────────────────────
    print("Building datasets...", flush=True)
    train_ds = AORCDataset(CFG, split="train")
    val_ds   = AORCDataset(CFG, split="val")

    loader_kwargs = {
        "num_workers":        CFG.num_workers,
        "pin_memory":         device.type == "cuda",
        "persistent_workers": CFG.num_workers > 0,
    }
    if CFG.num_workers > 0:
        loader_kwargs["prefetch_factor"] = CFG.prefetch_factor

    train_dl = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True, **loader_kwargs
    )
    val_dl = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False, **loader_kwargs
    )
    print(f"Train patches: {len(train_ds)}  Val patches: {len(val_ds)}", flush=True)

    # ── model ─────────────────────────────────────────────────────────────
    print("Building model...", flush=True)
    model = build_model().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if CFG.compile_model and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled with torch.compile (reduce-overhead)", flush=True)
        except Exception as exc:
            print(f"torch.compile unavailable ({exc}), continuing without.", flush=True)

    print("Model ready", flush=True)

    edm        = EDM(sigma_data=1.0)
    optimizer  = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    amp_dtype  = torch.bfloat16 if CFG.amp_dtype == "bf16" else torch.float16
    use_scaler = CFG.amp and amp_dtype == torch.float16
    scaler     = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ── tensorboard ───────────────────────────────────────────────────────
    tb_dir = out_dir / "tb_logs"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard logs → {tb_dir}", flush=True)
    print(f"  Launch with: tensorboard --logdir {tb_dir} --port 6006", flush=True)

    # ── resume ────────────────────────────────────────────────────────────
    start_epoch  = 0
    global_step  = 0
    ckpt_path    = out_dir / "last.pt"
    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", 0)
        except RuntimeError as exc:
            print(f"Checkpoint incompatible, starting from scratch. ({exc})", flush=True)

    # ── GPU monitor ───────────────────────────────────────────────────────
    gpu_id  = device.index if device.index is not None else 0
    monitor = GPUMonitor(gpu_id=gpu_id, interval=1.0, window=10)
    print(
        f"GPU monitor started (target util {CFG.gpu_util_low}–{CFG.gpu_util_high}%)",
        flush=True,
    )

    accum = CFG.grad_accum_steps

    # ── training ──────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, CFG.num_epochs):
            print(f"\nEpoch {epoch} start", flush=True)
            model.train()
            epoch_loss = 0.0
            t0         = time.time()
            last_log   = t0
            optimizer.zero_grad(set_to_none=True)

            for step, (cond, target) in enumerate(train_dl):
                cond   = cond.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last)
                target = target.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=CFG.amp):
                    loss = edm.loss(model, target, cond) / accum

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % accum == 0 or (step + 1) == len(train_dl):
                    if use_scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                step_loss   = loss.item() * accum
                epoch_loss += step_loss
                global_step += 1

                # log every step to TensorBoard, print every log_every steps
                writer.add_scalar("loss/train_step", step_loss, global_step)

                if step % CFG.log_every == 0:
                    now     = time.time()
                    secs    = now - last_log
                    sps     = (CFG.log_every * CFG.batch_size / secs) if step else 0.0
                    cur_util, avg_util, mem_used, mem_total = monitor.stats()
                    gpu_str = monitor.log_and_warn(step, CFG.gpu_util_low, CFG.gpu_util_high)

                    writer.add_scalar("gpu/utilization", cur_util,  global_step)
                    writer.add_scalar("gpu/vram_mib",    mem_used,  global_step)

                    print(
                        f"epoch {epoch:03d} step {step:05d}/{len(train_dl)} "
                        f"loss {step_loss:.6f} "
                        f"dt {secs:.1f}s sps {sps:.1f} | {gpu_str}",
                        flush=True,
                    )
                    last_log = now

            # ── validation ────────────────────────────────────────────────
            do_val = (
                epoch == start_epoch
                or (epoch + 1) % CFG.val_every == 0
                or (epoch + 1) == CFG.num_epochs
            )
            val_loss = None
            if do_val:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for cond, target in val_dl:
                        cond   = cond.to(device, non_blocking=True).contiguous(
                            memory_format=torch.channels_last)
                        target = target.to(device, non_blocking=True).contiguous(
                            memory_format=torch.channels_last)
                        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=CFG.amp):
                            val_loss += edm.loss(model, target, cond).item()
                val_loss /= len(val_dl)
                writer.add_scalar("loss/val_epoch", val_loss, epoch)

            train_loss = epoch_loss / len(train_dl)
            elapsed    = time.time() - t0
            val_text   = f"{val_loss:.6f}" if val_loss is not None else "skipped"
            _, avg_util, mem_used, mem_total = monitor.stats()

            writer.add_scalar("loss/train_epoch", train_loss, epoch)
            writer.add_scalar("gpu/avg_utilization_epoch", avg_util, epoch)

            print(
                f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_text} | "
                f"{elapsed:.1f}s | GPU avg {avg_util:.0f}% | "
                f"VRAM {mem_used}/{mem_total} MiB",
                flush=True,
            )

            # ── checkpoint ────────────────────────────────────────────────
            torch.save(
                {"epoch": epoch, "global_step": global_step,
                 "model": model.state_dict(),
                 "optimizer": optimizer.state_dict()},
                ckpt_path,
            )
            if (epoch + 1) % CFG.save_every == 0:
                torch.save(
                    {"epoch": epoch, "model": model.state_dict()},
                    out_dir / f"ckpt_epoch{epoch:03d}.pt",
                )

        writer.add_hparams(
            {"lr": CFG.lr, "batch_size": CFG.batch_size, "epochs": CFG.num_epochs,
             "base_ch": CFG.base_ch, "sigma_data": 1.0, "sampler": CFG.sampler},
            {"hparam/final_train_loss": train_loss,
             "hparam/final_val_loss": val_loss or float("nan")},
        )

    finally:
        writer.close()
        monitor.stop()


if __name__ == "__main__":
    train()
