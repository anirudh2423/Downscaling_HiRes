"""
Generate important evaluation plots for coarse-AORC-conditioned AORC temperature.

Run after training:
    python make_temperature_plots.py
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import CFG
from dataset import AORCDataset
from diffusion import EDM
from model import build_model


def denorm_temp(x, stats):
    return x * stats["temp_std"] + stats["temp_mean"]


def metrics(pred, truth):
    diff = pred - truth
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "corr": float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1]),
    }


def parse_loss_log(log_path):
    rows = []
    if not log_path.exists():
        return pd.DataFrame()
    pat = re.compile(
        r"Epoch\s+(\d+)\s+\|\s+train\s+([0-9.eE+-]+)\s+\|\s+val\s+([0-9.eE+-]+|skipped)"
    )
    for line in log_path.read_text(errors="ignore").splitlines():
        match = pat.search(line)
        if match:
            rows.append(
                {
                    "epoch": int(match.group(1)),
                    "train_loss": float(match.group(2)),
                    "val_loss": np.nan if match.group(3) == "skipped" else float(match.group(3)),
                }
            )
    return pd.DataFrame(rows)


def radial_power_spectrum(field):
    arr = field - np.mean(field)
    fft = np.fft.fftshift(np.fft.fft2(arr))
    power = np.abs(fft) ** 2
    y, x = np.indices(power.shape)
    r = np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2).astype(int)
    sums = np.bincount(r.ravel(), weights=power.ravel())
    counts = np.bincount(r.ravel())
    return sums / np.maximum(counts, 1)


def plot_loss_curves(out_dir):
    df = parse_loss_log(Path(CFG.output_dir) / "train.log")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], label="train")
    if df["val_loss"].notna().any():
        ax.plot(df["epoch"], df["val_loss"], label="validation", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("EDM loss")
    ax.set_title("Training vs validation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves_train_vs_validation.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def collect_predictions(n_samples=4):
    device = torch.device(CFG.device if torch.cuda.is_available() else "cpu")
    stats = np.load(CFG.stats_file)

    ds = AORCDataset(CFG, split="val")
    model = build_model().to(device)
    ckpt_path = Path(CFG.output_dir) / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    edm = EDM(sigma_data=1.0)
    rng = np.random.default_rng(CFG.seed)
    sample_ids = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    rows = []
    samples = []
    for sample_num, idx in enumerate(sample_ids):
        cond, target = ds[int(idx)]
        zarr_idx, t_idx, i, j = ds.index[int(idx)]
        half = CFG.patch_size // 2
        lat = ds.lat[i - half:i + half]
        lon = ds.lon[j - half:j + half]
        timestamp = str(ds.zarr_times[zarr_idx][t_idx])

        cond_b = cond.unsqueeze(0).to(device)
        target_b = target.unsqueeze(0).to(device)
        pred = edm.sample(
            model,
            cond_b,
            n_steps=CFG.sample_steps,
            n_target_channels=CFG.target_channels,
            device=str(device),
        )[0, 0].detach().cpu().numpy()

        sigma = torch.ones(1, 1, 1, 1, device=device)
        noisy = target_b + sigma * torch.randn_like(target_b)
        x_in = torch.cat([edm.c_in(sigma) * noisy, cond_b], dim=1)
        labels = edm.c_noise(torch.ones(1, device=device))
        f_x = model(x_in, labels).sample
        denoised = edm.c_skip(sigma) * noisy + edm.c_out(sigma) * f_x

        truth_k = denorm_temp(target[0].numpy(), stats)
        coarse_k = denorm_temp(cond[3].numpy(), stats)
        pred_k = denorm_temp(pred, stats)
        noisy_k = denorm_temp(noisy[0, 0].detach().cpu().numpy(), stats)
        denoised_k = denorm_temp(denoised[0, 0].detach().cpu().numpy(), stats)

        for name, arr in [
            ("coarse_aorc_baseline", coarse_k),
            ("diffusion", pred_k),
            ("noisy_sigma1", noisy_k),
            ("edm_denoised_sigma1", denoised_k),
        ]:
            row = {"sample": sample_num, "source": name, "time": timestamp}
            row.update(metrics(arr, truth_k))
            rows.append(row)

        samples.append(
            {
                "sample": sample_num,
                "time": timestamp,
                "lat": lat,
                "lon": lon,
                "truth": truth_k,
                "coarse": coarse_k,
                "pred": pred_k,
                "noisy": noisy_k,
                "denoised": denoised_k,
            }
        )

    return samples, pd.DataFrame(rows)


def plot_map_panels(samples, out_dir):
    fig, axes = plt.subplots(len(samples), 5, figsize=(18, 3.5 * len(samples)), squeeze=False)
    for r, sample in enumerate(samples):
        truth = sample["truth"]
        coarse = sample["coarse"]
        pred = sample["pred"]
        extent = [sample["lon"][0], sample["lon"][-1], sample["lat"][0], sample["lat"][-1]]
        vmin = min(np.nanpercentile(truth, 2), np.nanpercentile(coarse, 2), np.nanpercentile(pred, 2))
        vmax = max(np.nanpercentile(truth, 98), np.nanpercentile(coarse, 98), np.nanpercentile(pred, 98))
        fields = [
            ("AORC temp", truth, "turbo", None),
            ("Coarse AORC cond", coarse, "turbo", None),
            ("Diffusion", pred, "turbo", None),
            ("Diffusion - AORC", pred - truth, "RdBu_r", (-5, 5)),
            ("Coarse - AORC", coarse - truth, "RdBu_r", (-5, 5)),
        ]
        for c, (title, arr, cmap, lim) in enumerate(fields):
            ax = axes[r, c]
            if lim is None:
                im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_title(f"Sample {sample['sample']} | {title}\n{sample['time']}", fontsize=9)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("K")
    fig.tight_layout()
    fig.savefig(out_dir / "us_map_aorc_coarse_diffusion_temperature.png", dpi=180)
    plt.close(fig)


def plot_denoising_panels(samples, out_dir):
    fig, axes = plt.subplots(len(samples), 4, figsize=(15, 3.5 * len(samples)), squeeze=False)
    for r, sample in enumerate(samples):
        truth = sample["truth"]
        noisy = sample["noisy"]
        denoised = sample["denoised"]
        extent = [sample["lon"][0], sample["lon"][-1], sample["lat"][0], sample["lat"][-1]]
        vmin = min(np.nanpercentile(truth, 2), np.nanpercentile(noisy, 2), np.nanpercentile(denoised, 2))
        vmax = max(np.nanpercentile(truth, 98), np.nanpercentile(noisy, 98), np.nanpercentile(denoised, 98))
        fields = [
            ("AORC temp", truth, "turbo", None),
            ("Noisy input sigma=1", noisy, "turbo", None),
            ("EDM denoised", denoised, "turbo", None),
            ("Denoised - AORC", denoised - truth, "RdBu_r", (-5, 5)),
        ]
        for c, (title, arr, cmap, lim) in enumerate(fields):
            ax = axes[r, c]
            if lim is None:
                im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_title(f"Sample {sample['sample']} | {title}\n{sample['time']}", fontsize=9)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("K")
    fig.tight_layout()
    fig.savefig(out_dir / "denoising_temperature_sigma1_panels.png", dpi=180)
    plt.close(fig)


def plot_sample_locations(samples, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24, 50)
    ax.set_title("Validation sample locations over AORC CONUS domain")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    for sample in samples:
        lon0, lon1 = float(sample["lon"][0]), float(sample["lon"][-1])
        lat0, lat1 = float(sample["lat"][0]), float(sample["lat"][-1])
        ax.add_patch(
            plt.Rectangle(
                (min(lon0, lon1), min(lat0, lat1)),
                abs(lon1 - lon0),
                abs(lat1 - lat0),
                fill=False,
                linewidth=1.5,
            )
        )
        ax.text((lon0 + lon1) / 2, (lat0 + lat1) / 2, str(sample["sample"]),
                ha="center", va="center", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "us_domain_validation_sample_locations.png", dpi=180)
    plt.close(fig)


def plot_metrics_summary(metrics_df, out_dir):
    keep = metrics_df[metrics_df["source"].isin(["coarse_aorc_baseline", "diffusion"])]
    summary = keep.groupby("source")[["rmse", "mae", "bias", "corr"]].mean()
    summary.to_csv(out_dir / "metrics_summary_mean.csv")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, metric in zip(axes, ["rmse", "mae", "bias", "corr"]):
        vals = summary[metric].reindex(["coarse_aorc_baseline", "diffusion"])
        ax.bar(["Coarse AORC", "Diffusion"], vals.values, color=["#777777", "#2563eb"])
        ax.set_title(metric.upper())
        ax.grid(axis="y", alpha=0.25)
        if metric in {"rmse", "mae", "bias"}:
            ax.set_ylabel("K")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_summary_coarse_aorc_vs_diffusion.png", dpi=180)
    plt.close(fig)


def plot_scatter_and_distributions(samples, out_dir):
    truth = np.concatenate([s["truth"].ravel() for s in samples])
    coarse = np.concatenate([s["coarse"].ravel() for s in samples])
    pred = np.concatenate([s["pred"].ravel() for s in samples])
    rng = np.random.default_rng(CFG.seed)
    take = rng.choice(len(truth), size=min(200000, len(truth)), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(truth, bins=80, alpha=0.55, label="AORC")
    axes[0].hist(coarse, bins=80, alpha=0.55, label="Coarse AORC")
    axes[0].hist(pred, bins=80, alpha=0.55, label="Diffusion")
    axes[0].set_title("Temperature distributions")
    axes[0].set_xlabel("K")
    axes[0].legend()

    axes[1].scatter(truth[take], coarse[take], s=1, alpha=0.08)
    axes[1].set_title("AORC vs coarse baseline")
    axes[1].set_xlabel("AORC K")
    axes[1].set_ylabel("Coarse AORC K")

    axes[2].scatter(truth[take], pred[take], s=1, alpha=0.08)
    axes[2].set_title("AORC vs diffusion")
    axes[2].set_xlabel("AORC K")
    axes[2].set_ylabel("Diffusion K")

    lo, hi = np.nanpercentile(truth, [1, 99])
    for ax in axes[1:]:
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "temperature_distributions_and_scatter.png", dpi=180)
    plt.close(fig)


def plot_power_spectrum(samples, out_dir):
    spectra = {"AORC": [], "Coarse AORC": [], "Diffusion": []}
    for sample in samples:
        spectra["AORC"].append(radial_power_spectrum(sample["truth"]))
        spectra["Coarse AORC"].append(radial_power_spectrum(sample["coarse"]))
        spectra["Diffusion"].append(radial_power_spectrum(sample["pred"]))

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, vals in spectra.items():
        n = min(len(v) for v in vals)
        arr = np.stack([v[:n] for v in vals])
        ax.loglog(np.arange(n), arr.mean(axis=0), label=name)
    ax.set_title("Radial power spectrum")
    ax.set_xlabel("Spatial frequency bin")
    ax.set_ylabel("Power")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "temperature_power_spectrum.png", dpi=180)
    plt.close(fig)


def main():
    out_dir = Path(CFG.plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(out_dir)
    samples, metrics_df = collect_predictions()
    metrics_df.to_csv(out_dir / "sample_metrics_coarse_aorc_vs_diffusion.csv", index=False)

    np.savez_compressed(
        out_dir / "sample_predictions.npz",
        **{f"sample_{s['sample']}_truth": s["truth"] for s in samples},
        **{f"sample_{s['sample']}_coarse": s["coarse"] for s in samples},
        **{f"sample_{s['sample']}_diffusion": s["pred"] for s in samples},
    )
    plot_sample_locations(samples, out_dir)
    plot_metrics_summary(metrics_df, out_dir)
    plot_map_panels(samples, out_dir)
    plot_denoising_panels(samples, out_dir)
    plot_scatter_and_distributions(samples, out_dir)
    plot_power_spectrum(samples, out_dir)
    print(f"Saved plots and metrics -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
