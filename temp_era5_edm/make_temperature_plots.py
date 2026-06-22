"""
Generate evaluation plots for the ERA5-conditioned AORC temperature model.

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


def denorm_target(x, stats):
    return x * stats["temp_std"] + stats["temp_mean"]


def denorm_era5(x, stats):
    return x * stats["era5_t2m_std"] + stats["era5_t2m_mean"]


def metrics(pred, truth):
    diff = pred - truth
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    corr = float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1])
    return {"rmse": rmse, "mae": mae, "bias": bias, "corr": corr}


def parse_loss_log(log_path):
    rows = []
    if not log_path.exists():
        return pd.DataFrame()
    pat = re.compile(
        r"Epoch\s+(\d+)\s+\|\s+train\s+([0-9.eE+-]+)\s+\|\s+val\s+([0-9.eE+-]+|skipped)"
    )
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if not m:
            continue
        epoch = int(m.group(1))
        train = float(m.group(2))
        val = np.nan if m.group(3) == "skipped" else float(m.group(3))
        rows.append({"epoch": epoch, "train_loss": train, "val_loss": val})
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
        eps = torch.randn_like(target_b)
        noisy = target_b + sigma * eps
        x_in = torch.cat([edm.c_in(sigma) * noisy, cond_b], dim=1)
        labels = edm.c_noise(torch.ones(1, device=device))
        f_x = model(x_in, labels).sample
        denoised = edm.c_skip(sigma) * noisy + edm.c_out(sigma) * f_x

        truth_k = denorm_target(target[0].numpy(), stats)
        pred_k = denorm_target(pred, stats)
        era5_k = denorm_era5(cond[3].numpy(), stats)
        noisy_k = denorm_target(noisy[0, 0].detach().cpu().numpy(), stats)
        denoised_k = denorm_target(denoised[0, 0].detach().cpu().numpy(), stats)

        for name, arr in [
            ("era5_baseline", era5_k),
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
                "era5": era5_k,
                "pred": pred_k,
                "noisy": noisy_k,
                "denoised": denoised_k,
            }
        )

    return samples, pd.DataFrame(rows)


def plot_panels(samples, out_dir):
    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(18, 3.5 * n), squeeze=False)
    for r, s in enumerate(samples):
        truth = s["truth"]
        era5 = s["era5"]
        pred = s["pred"]
        fields = [
            ("AORC temp", truth, "turbo", None),
            ("ERA5 interp", era5, "turbo", None),
            ("Diffusion", pred, "turbo", None),
            ("Diffusion - AORC", pred - truth, "RdBu_r", (-5, 5)),
            ("ERA5 - AORC", era5 - truth, "RdBu_r", (-5, 5)),
        ]
        extent = [s["lon"][0], s["lon"][-1], s["lat"][0], s["lat"][-1]]
        vmin = min(np.nanpercentile(truth, 2), np.nanpercentile(era5, 2), np.nanpercentile(pred, 2))
        vmax = max(np.nanpercentile(truth, 98), np.nanpercentile(era5, 98), np.nanpercentile(pred, 98))
        for c, (title, arr, cmap, lim) in enumerate(fields):
            ax = axes[r, c]
            if lim is None:
                im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_title(f"Sample {s['sample']} | {title}\n{s['time']}", fontsize=9)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("K")
    fig.tight_layout()
    fig.savefig(out_dir / "aorc_era5_diffusion_temperature_panels.png", dpi=180)
    plt.close(fig)


def plot_denoising_panels(samples, out_dir):
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.5 * n), squeeze=False)
    for r, s in enumerate(samples):
        truth = s["truth"]
        noisy = s["noisy"]
        denoised = s["denoised"]
        extent = [s["lon"][0], s["lon"][-1], s["lat"][0], s["lat"][-1]]
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
            ax.set_title(f"Sample {s['sample']} | {title}\n{s['time']}", fontsize=9)
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
    ax.set_title("Validation sample locations over ERA5 USA domain")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    for s in samples:
        lon0, lon1 = float(s["lon"][0]), float(s["lon"][-1])
        lat0, lat1 = float(s["lat"][0]), float(s["lat"][-1])
        ax.add_patch(
            plt.Rectangle(
                (min(lon0, lon1), min(lat0, lat1)),
                abs(lon1 - lon0),
                abs(lat1 - lat0),
                fill=False,
                linewidth=1.5,
            )
        )
        ax.text((lon0 + lon1) / 2, (lat0 + lat1) / 2, str(s["sample"]),
                ha="center", va="center", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "us_domain_validation_sample_locations.png", dpi=180)
    plt.close(fig)


def plot_metrics_summary(metrics_df, out_dir):
    keep = metrics_df[metrics_df["source"].isin(["era5_baseline", "diffusion"])]
    summary = keep.groupby("source")[["rmse", "mae", "bias", "corr"]].mean()
    summary.to_csv(out_dir / "metrics_summary_mean.csv")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, metric in zip(axes, ["rmse", "mae", "bias", "corr"]):
        vals = summary[metric].reindex(["era5_baseline", "diffusion"])
        ax.bar(["ERA5", "Diffusion"], vals.values, color=["#777777", "#2563eb"])
        ax.set_title(metric.upper())
        ax.grid(axis="y", alpha=0.25)
        if metric in {"rmse", "mae", "bias"}:
            ax.set_ylabel("K")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_summary_era5_vs_diffusion.png", dpi=180)
    plt.close(fig)


def plot_scatter_and_distributions(samples, out_dir):
    truth = np.concatenate([s["truth"].ravel() for s in samples])
    era5 = np.concatenate([s["era5"].ravel() for s in samples])
    pred = np.concatenate([s["pred"].ravel() for s in samples])
    rng = np.random.default_rng(CFG.seed)
    take = rng.choice(len(truth), size=min(200000, len(truth)), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(truth, bins=80, alpha=0.55, label="AORC")
    axes[0].hist(era5, bins=80, alpha=0.55, label="ERA5")
    axes[0].hist(pred, bins=80, alpha=0.55, label="Diffusion")
    axes[0].set_title("Temperature distributions")
    axes[0].set_xlabel("K")
    axes[0].legend()

    axes[1].scatter(truth[take], era5[take], s=1, alpha=0.08)
    axes[1].set_title("AORC vs ERA5 baseline")
    axes[1].set_xlabel("AORC K")
    axes[1].set_ylabel("ERA5 K")

    axes[2].scatter(truth[take], pred[take], s=1, alpha=0.08)
    axes[2].set_title("AORC vs diffusion")
    axes[2].set_xlabel("AORC K")
    axes[2].set_ylabel("Diffusion K")

    for ax in axes[1:]:
        lo, hi = np.nanpercentile(truth, [1, 99])
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "temperature_distributions_and_scatter.png", dpi=180)
    plt.close(fig)


def plot_power_spectrum(samples, out_dir):
    spectra = {"AORC": [], "ERA5": [], "Diffusion": []}
    for s in samples:
        spectra["AORC"].append(radial_power_spectrum(s["truth"]))
        spectra["ERA5"].append(radial_power_spectrum(s["era5"]))
        spectra["Diffusion"].append(radial_power_spectrum(s["pred"]))

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
    metrics_df.to_csv(out_dir / "sample_metrics_era5_vs_diffusion.csv", index=False)
    plot_sample_locations(samples, out_dir)
    plot_metrics_summary(metrics_df, out_dir)
    plot_panels(samples, out_dir)
    plot_denoising_panels(samples, out_dir)
    plot_scatter_and_distributions(samples, out_dir)
    plot_power_spectrum(samples, out_dir)
    print(f"Saved plots and metrics -> {out_dir}")


if __name__ == "__main__":
    main()
