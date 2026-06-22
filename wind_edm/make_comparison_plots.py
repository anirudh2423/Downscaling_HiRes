import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CFG
from dataset import AORCDataset
from diffusion import EDM
from model import build_model


EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)\s+\|\s+train\s+([0-9.]+)\s+\|\s+val\s+([0-9.]+|skipped)\s+\|\s+"
    r"([0-9.]+)s\s+\|\s+GPU avg\s+([0-9.]+)%\s+\|\s+VRAM\s+([0-9]+)/([0-9]+)"
)
STEP_RE = re.compile(
    r"^epoch\s+(\d+)\s+step\s+(\d+)/(\d+)\s+loss\s+([0-9.]+)\s+dt\s+([0-9.]+)s\s+sps\s+([0-9.]+)"
)


def parse_training_log(path):
    epochs = []
    steps = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                epoch = int(m.group(1))
                val = np.nan if m.group(3) == "skipped" else float(m.group(3))
                epochs.append(
                    {
                        "epoch": epoch,
                        "train": float(m.group(2)),
                        "val": val,
                        "seconds": float(m.group(4)),
                        "gpu_avg": float(m.group(5)),
                        "vram_mib": float(m.group(6)),
                        "vram_total_mib": float(m.group(7)),
                    }
                )
                continue
            m = STEP_RE.search(line)
            if m:
                steps.append(
                    {
                        "epoch": int(m.group(1)),
                        "step": int(m.group(2)),
                        "step_total": int(m.group(3)),
                        "loss": float(m.group(4)),
                        "dt": float(m.group(5)),
                        "sps": float(m.group(6)),
                    }
                )
    return epochs, steps


def save_training_curves(epochs, steps, out_dir):
    ep = np.array([e["epoch"] for e in epochs])
    train = np.array([e["train"] for e in epochs])
    val = np.array([e["val"] for e in epochs])

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.plot(ep, train, marker="o", lw=2.2, label="Train loss")
    valid = ~np.isnan(val)
    ax.plot(ep[valid], val[valid], marker="s", lw=2.2, label="Validation loss")
    ax.set_title("EDM Wind Training Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("EDM loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves_train_vs_validation.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=160)
    axes = axes.ravel()
    axes[0].plot(ep, [e["seconds"] / 60 for e in epochs], marker="o", color="#2a6f97")
    axes[0].set_title("Epoch Duration")
    axes[0].set_ylabel("Minutes")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(ep, [e["gpu_avg"] for e in epochs], marker="o", color="#00876c")
    axes[1].set_title("Average GPU Utilization")
    axes[1].set_ylabel("Percent")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(ep, [e["vram_mib"] / 1024 for e in epochs], marker="o", color="#7b2cbf")
    axes[2].set_title("VRAM at Epoch Summary")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("GiB")
    axes[2].grid(True, alpha=0.25)

    if steps:
        x = np.array([s["epoch"] + s["step"] / max(s["step_total"], 1) for s in steps])
        sps = np.array([s["sps"] for s in steps])
        axes[3].plot(x[sps > 0], sps[sps > 0], lw=1.3, color="#c2410c")
    axes[3].set_title("Throughput")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Samples/sec")
    axes[3].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "training_diagnostics.png")
    plt.close(fig)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {
        (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
        for k, v in state.items()
    }
    model = build_model()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing={missing}, unexpected={unexpected}")
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    model.eval()
    return model


def denormalize(target_norm, ds):
    target = np.empty_like(target_norm, dtype=np.float32)
    target[:, 0] = target_norm[:, 0] * ds.ugrd_std + ds.ugrd_mean
    target[:, 1] = target_norm[:, 1] * ds.vgrd_std + ds.vgrd_mean
    return target


def speed(arr):
    return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def corr(a, b):
    af = a.ravel()
    bf = b.ravel()
    if np.std(af) == 0 or np.std(bf) == 0:
        return float("nan")
    return float(np.corrcoef(af, bf)[0, 1])


def radial_power_spectrum(field):
    arr = field - np.nanmean(field)
    power = np.abs(np.fft.rfft2(arr)) ** 2
    yy, xx = np.indices(power.shape)
    rr = np.sqrt(yy**2 + xx**2).astype(np.int32)
    counts = np.bincount(rr.ravel())
    sums = np.bincount(rr.ravel(), weights=power.ravel())
    valid = counts > 0
    return np.arange(len(counts))[valid], sums[valid] / counts[valid]


def finite_limits(*arrays, q=(1, 99)):
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    return np.percentile(vals, q)


def sample_predictions(model, ds, sample_indices, n_steps, device, seed):
    torch.manual_seed(seed)
    conds = []
    targets = []
    metadata = []
    for idx in sample_indices:
        cond, target = ds[int(idx)]
        conds.append(cond)
        targets.append(target)
        zarr_idx, t_idx, i, j = ds.index[int(idx)]
        metadata.append((zarr_idx, t_idx, i, j))

    cond = torch.stack(conds).to(device)
    target_norm = torch.stack(targets).numpy()
    if device.type == "cuda":
        cond = cond.contiguous(memory_format=torch.channels_last)

    edm = EDM(sigma_data=1.0)
    preds = []
    with torch.no_grad():
        for n in range(cond.shape[0]):
            c = cond[n : n + 1]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = edm.sample(model, c, n_steps=n_steps, n_target_channels=2, device=device)
            preds.append(pred.float().cpu())
    pred_norm = torch.cat(preds, dim=0).numpy()
    return cond.cpu().numpy(), target_norm, pred_norm, metadata


def denoise_predictions(model, cond_np, target_norm_np, sigma_value, device, seed):
    torch.manual_seed(seed)
    cond = torch.from_numpy(cond_np).to(device)
    target = torch.from_numpy(target_norm_np).to(device)
    if device.type == "cuda":
        cond = cond.contiguous(memory_format=torch.channels_last)
        target = target.contiguous(memory_format=torch.channels_last)

    edm = EDM(sigma_data=1.0)
    sigma = torch.full((target.shape[0], 1, 1, 1), float(sigma_value), device=device)
    eps = torch.randn_like(target)
    noisy = target + sigma * eps

    with torch.no_grad():
        x_in = torch.cat([edm.c_in(sigma) * noisy, cond], dim=1)
        noise_labels = edm.c_noise(sigma.view(target.shape[0]))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            f_x = model(x_in, noise_labels).sample
        denoised = edm.c_skip(sigma) * noisy + edm.c_out(sigma) * f_x.float()
    return noisy.float().cpu().numpy(), denoised.float().cpu().numpy()


def save_sample_comparison(k, cond, truth, pred, metadata, out_dir):
    topo, svf, csza = cond[:3]
    coarse_u = cond[3] if cond.shape[0] > 3 else None
    coarse_v = cond[4] if cond.shape[0] > 4 else None
    truth_speed = np.sqrt(truth[0] ** 2 + truth[1] ** 2)
    pred_speed = np.sqrt(pred[0] ** 2 + pred[1] ** 2)
    speed_err = pred_speed - truth_speed

    wind_vmin, wind_vmax = finite_limits(truth_speed, pred_speed)
    uv_vmin, uv_vmax = finite_limits(truth[0], pred[0], truth[1], pred[1], q=(2, 98))
    err_abs = max(
        np.percentile(np.abs(pred[0] - truth[0]), 98),
        np.percentile(np.abs(pred[1] - truth[1]), 98),
        np.percentile(np.abs(speed_err), 98),
        1e-3,
    )

    panels = [
        (topo, "Input topography, normalized", "terrain", None, None),
        (svf, "Input sky-view factor", "viridis", 0, 1),
        (csza, "Input cos solar zenith", "magma", 0, 1),
        (truth_speed, "AORC wind speed", "turbo", wind_vmin, wind_vmax),
        (pred_speed, "Diffusion wind speed", "turbo", wind_vmin, wind_vmax),
        (speed_err, "Speed error: generated - AORC", "coolwarm", -err_abs, err_abs),
        (truth[0], "AORC U wind", "RdBu_r", uv_vmin, uv_vmax),
        (pred[0], "Diffusion U wind", "RdBu_r", uv_vmin, uv_vmax),
        (pred[0] - truth[0], "U error", "coolwarm", -err_abs, err_abs),
        (truth[1], "AORC V wind", "RdBu_r", uv_vmin, uv_vmax),
        (pred[1], "Diffusion V wind", "RdBu_r", uv_vmin, uv_vmax),
        (pred[1] - truth[1], "V error", "coolwarm", -err_abs, err_abs),
    ]
    if coarse_u is not None and coarse_v is not None:
        panels[0] = (coarse_u, "Input coarse U, normalized", "RdBu_r", None, None)
        panels[1] = (coarse_v, "Input coarse V, normalized", "RdBu_r", None, None)
        panels[2] = (topo, "Input topography, normalized", "terrain", None, None)

    fig, axes = plt.subplots(4, 3, figsize=(13, 15), dpi=150)
    zarr_idx, t_idx, i, j = metadata
    fig.suptitle(
        f"Sample {k:02d}: zarr={zarr_idx}, time_index={t_idx}, center=({i}, {j})",
        y=0.995,
        fontsize=14,
    )
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_dir / f"comparison_inputs_aorc_vs_diffusion_sample_{k:02d}.png")
    plt.close(fig)


def save_quiver(k, truth, pred, out_dir):
    stride = 16
    yy, xx = np.mgrid[0 : truth.shape[1] : stride, 0 : truth.shape[2] : stride]
    truth_speed = np.sqrt(truth[0] ** 2 + truth[1] ** 2)
    pred_speed = np.sqrt(pred[0] ** 2 + pred[1] ** 2)
    wind_vmin, wind_vmax = finite_limits(truth_speed, pred_speed)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=160)
    for ax, arr, title in [
        (axes[0], truth, "AORC vector field"),
        (axes[1], pred, "Diffusion vector field"),
    ]:
        spd = np.sqrt(arr[0] ** 2 + arr[1] ** 2)
        im = ax.imshow(spd, cmap="turbo", vmin=wind_vmin, vmax=wind_vmax, origin="lower")
        ax.quiver(
            xx,
            yy,
            arr[0, ::stride, ::stride],
            arr[1, ::stride, ::stride],
            color="black",
            alpha=0.65,
            width=0.0022,
            scale=450,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="m/s")
    fig.tight_layout()
    fig.savefig(out_dir / f"quiver_aorc_vs_diffusion_sample_{k:02d}.png")
    plt.close(fig)


def save_denoising_comparison(k, truth, noisy, denoised, sigma_value, out_dir):
    truth_speed = np.sqrt(truth[0] ** 2 + truth[1] ** 2)
    noisy_speed = np.sqrt(noisy[0] ** 2 + noisy[1] ** 2)
    denoised_speed = np.sqrt(denoised[0] ** 2 + denoised[1] ** 2)
    wind_vmin, wind_vmax = finite_limits(truth_speed, noisy_speed, denoised_speed)
    uv_vmin, uv_vmax = finite_limits(truth[0], noisy[0], denoised[0], truth[1], noisy[1], denoised[1], q=(2, 98))
    err_abs = max(
        np.percentile(np.abs(noisy[0] - truth[0]), 98),
        np.percentile(np.abs(denoised[0] - truth[0]), 98),
        np.percentile(np.abs(noisy[1] - truth[1]), 98),
        np.percentile(np.abs(denoised[1] - truth[1]), 98),
        1e-3,
    )

    panels = [
        (truth_speed, "AORC speed", "turbo", wind_vmin, wind_vmax),
        (noisy_speed, f"Noisy diffusion input speed, sigma={sigma_value}", "turbo", wind_vmin, wind_vmax),
        (denoised_speed, "EDM denoised speed", "turbo", wind_vmin, wind_vmax),
        (denoised_speed - truth_speed, "Denoised speed error", "coolwarm", -err_abs, err_abs),
        (truth[0], "AORC U", "RdBu_r", uv_vmin, uv_vmax),
        (noisy[0], "Noisy U input", "RdBu_r", uv_vmin, uv_vmax),
        (denoised[0], "Denoised U", "RdBu_r", uv_vmin, uv_vmax),
        (denoised[0] - truth[0], "Denoised U error", "coolwarm", -err_abs, err_abs),
        (truth[1], "AORC V", "RdBu_r", uv_vmin, uv_vmax),
        (noisy[1], "Noisy V input", "RdBu_r", uv_vmin, uv_vmax),
        (denoised[1], "Denoised V", "RdBu_r", uv_vmin, uv_vmax),
        (denoised[1] - truth[1], "Denoised V error", "coolwarm", -err_abs, err_abs),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), dpi=150)
    fig.suptitle(f"EDM Denoising Diagnostic: Sample {k:02d}", y=0.995, fontsize=14)
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    tag = str(sigma_value).replace(".", "p")
    fig.savefig(out_dir / f"denoising_noisy_input_vs_edm_sigma_{tag}_sample_{k:02d}.png")
    plt.close(fig)


def save_distribution_plots(truth, pred, out_dir):
    true_speed = speed(truth)
    pred_speed = speed(pred)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=160)
    items = [
        (truth[:, 0], pred[:, 0], "U wind (m/s)"),
        (truth[:, 1], pred[:, 1], "V wind (m/s)"),
        (true_speed, pred_speed, "Wind speed (m/s)"),
    ]
    for ax, (a, b, title) in zip(axes[0], items):
        lo, hi = finite_limits(a, b, q=(0.5, 99.5))
        bins = np.linspace(lo, hi, 80)
        ax.hist(a.ravel(), bins=bins, histtype="step", lw=2, density=True, label="AORC")
        ax.hist(b.ravel(), bins=bins, histtype="step", lw=2, density=True, label="Diffusion")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.legend()
    for ax, (a, b, title) in zip(axes[1], items):
        rng = np.random.default_rng(0)
        n = min(50000, a.size)
        idx = rng.choice(a.size, n, replace=False)
        af = a.ravel()[idx]
        bf = b.ravel()[idx]
        lo, hi = finite_limits(af, bf, q=(0.5, 99.5))
        ax.hexbin(af, bf, gridsize=80, bins="log", cmap="viridis", mincnt=1)
        ax.plot([lo, hi], [lo, hi], color="white", lw=1.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("AORC")
        ax.set_ylabel("Diffusion")
        ax.set_title(f"{title}: generated vs true")
        ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_dir / "variable_distributions_and_scatter.png")
    plt.close(fig)


def save_metrics(truth, pred, out_dir):
    true_speed = speed(truth)
    pred_speed = speed(pred)
    rows = []
    for name, a, b in [
        ("U", truth[:, 0], pred[:, 0]),
        ("V", truth[:, 1], pred[:, 1]),
        ("speed", true_speed, pred_speed),
    ]:
        rows.append(
            {
                "variable": name,
                "rmse": rmse(a, b),
                "mae": mae(a, b),
                "bias": float(np.mean(b - a)),
                "corr": corr(a, b),
                "true_mean": float(np.mean(a)),
                "pred_mean": float(np.mean(b)),
                "true_std": float(np.std(a)),
                "pred_std": float(np.std(b)),
            }
        )

    csv_path = out_dir / "sample_metrics_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")

    labels = [r["variable"] for r in rows]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.bar(x - width, [r["rmse"] for r in rows], width, label="RMSE")
    ax.bar(x, [r["mae"] for r in rows], width, label="MAE")
    ax.bar(x + width, [abs(r["bias"]) for r in rows], width, label="|Bias|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("m/s")
    ax.set_title("Sample-Level Error Metrics")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_rmse_mae_bias.png")
    plt.close(fig)

    return rows


def save_denoising_metrics(truth, noisy, denoised, sigma_value, out_dir):
    rows = []
    for source_name, arr in [("noisy_input", noisy), ("edm_denoised", denoised)]:
        for name, a, b in [
            ("U", truth[:, 0], arr[:, 0]),
            ("V", truth[:, 1], arr[:, 1]),
            ("speed", speed(truth), speed(arr)),
        ]:
            rows.append(
                {
                    "source": source_name,
                    "sigma": sigma_value,
                    "variable": name,
                    "rmse": rmse(a, b),
                    "mae": mae(a, b),
                    "bias": float(np.mean(b - a)),
                    "corr": corr(a, b),
                }
            )

    tag = str(sigma_value).replace(".", "p")
    csv_path = out_dir / f"denoising_metrics_sigma_{tag}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")

    labels = ["U", "V", "speed"]
    x = np.arange(len(labels))
    width = 0.35
    noisy_rmse = [r["rmse"] for r in rows if r["source"] == "noisy_input"]
    den_rmse = [r["rmse"] for r in rows if r["source"] == "edm_denoised"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.bar(x - width / 2, noisy_rmse, width, label="Noisy input")
    ax.bar(x + width / 2, den_rmse, width, label="EDM denoised")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title(f"Denoising RMSE Improvement, sigma={sigma_value}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"denoising_rmse_improvement_sigma_{tag}.png")
    plt.close(fig)
    return rows


def save_spectra(truth, pred, out_dir):
    true_speed = speed(truth)
    pred_speed = speed(pred)
    spectra_true = []
    spectra_pred = []
    for i in range(truth.shape[0]):
        k_true, p_true = radial_power_spectrum(true_speed[i])
        k_pred, p_pred = radial_power_spectrum(pred_speed[i])
        n = min(len(k_true), len(k_pred))
        spectra_true.append(p_true[:n])
        spectra_pred.append(p_pred[:n])
    n = min(min(map(len, spectra_true)), min(map(len, spectra_pred)))
    p_true = np.mean([p[:n] for p in spectra_true], axis=0)
    p_pred = np.mean([p[:n] for p in spectra_pred], axis=0)
    kvals = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.loglog(kvals[1:], p_true[1:], lw=2, label="AORC speed")
    ax.loglog(kvals[1:], p_pred[1:], lw=2, label="Diffusion speed")
    ax.set_xlabel("Radial wavenumber")
    ax.set_ylabel("Power")
    ax.set_title("Wind-Speed Spatial Power Spectrum")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "wind_speed_power_spectrum.png")
    plt.close(fig)


def save_conditioning_summary(cond, out_dir):
    titles = [
        "Topography normalized",
        "Sky-view factor",
        "Cos solar zenith",
        "Coarse U normalized",
        "Coarse V normalized",
    ][: cond.shape[1]]
    fig, axes = plt.subplots(1, len(titles), figsize=(4.2 * len(titles), 4), dpi=160)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        vals = cond[:, i].ravel()
        ax.hist(vals, bins=80, color="#2a6f97", alpha=0.9)
        ax.set_title(titles[i])
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "conditioning_input_distributions.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(Path(CFG.output_dir) / "last.pt"))
    parser.add_argument("--log", default=str(Path(CFG.output_dir) / "train.log"))
    parser.add_argument("--out_dir", default=str(Path(CFG.output_dir) / "plots"))
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--denoise_sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=CFG.device)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs, steps = parse_training_log(args.log)
    if not epochs:
        raise RuntimeError(f"No epoch summaries found in {args.log}")
    save_training_curves(epochs, steps, out_dir)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("Building validation dataset...")
    ds = AORCDataset(CFG, split="val")
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)
    print(f"Selected validation indices: {sample_indices.tolist()}")

    print("Loading checkpoint...")
    model = load_model(args.ckpt, device)

    print(f"Sampling {len(sample_indices)} patches with {args.n_steps} EDM steps...")
    cond, truth_norm, pred_norm, metadata = sample_predictions(
        model, ds, sample_indices, args.n_steps, device, args.seed
    )
    truth = denormalize(truth_norm, ds)
    pred = denormalize(pred_norm, ds)
    noisy_norm, denoised_norm = denoise_predictions(
        model, cond, truth_norm, args.denoise_sigma, device, args.seed + 1
    )
    noisy = denormalize(noisy_norm, ds)
    denoised = denormalize(denoised_norm, ds)

    np.savez_compressed(
        out_dir / "sample_predictions.npz",
        cond=cond,
        truth=truth,
        pred=pred,
        noisy=noisy,
        denoised=denoised,
        sample_indices=sample_indices,
        metadata=np.array(metadata, dtype=np.int64),
    )

    save_conditioning_summary(cond, out_dir)
    for k in range(truth.shape[0]):
        save_sample_comparison(k, cond[k], truth[k], pred[k], metadata[k], out_dir)
        save_quiver(k, truth[k], pred[k], out_dir)
        save_denoising_comparison(k, truth[k], noisy[k], denoised[k], args.denoise_sigma, out_dir)

    save_distribution_plots(truth, pred, out_dir)
    rows = save_metrics(truth, pred, out_dir)
    denoising_rows = save_denoising_metrics(truth, noisy, denoised, args.denoise_sigma, out_dir)
    save_spectra(truth, pred, out_dir)

    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("EDM wind comparison plot bundle\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Sampler steps: {args.n_steps}\n")
        f.write(f"Sample indices: {sample_indices.tolist()}\n")
        f.write("Metrics:\n")
        for row in rows:
            f.write(
                f"  {row['variable']}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, "
                f"bias={row['bias']:.4f}, corr={row['corr']:.4f}\n"
            )
        f.write(f"Denoising metrics, sigma={args.denoise_sigma}:\n")
        for row in denoising_rows:
            f.write(
                f"  {row['source']} {row['variable']}: RMSE={row['rmse']:.4f}, "
                f"MAE={row['mae']:.4f}, bias={row['bias']:.4f}, corr={row['corr']:.4f}\n"
            )
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
