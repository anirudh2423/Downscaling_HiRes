"""
Make a wind-style comparison panel for temperature.

Temperature is plotted as local anomaly (field minus patch mean) so the spatial
structure is visible on a centered red/blue scale, similar to U/V wind panels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CFG
from dataset import AORCDataset
from diffusion import EDM
from model import build_model


def denorm_temp(x, stats):
    return x * stats["temp_std"] + stats["temp_mean"]


def anomaly(x):
    return x - np.nanmean(x)


@torch.no_grad()
def collect_samples(n_samples=4):
    device = torch.device(CFG.device if torch.cuda.is_available() else "cpu")
    stats = np.load(CFG.stats_file)

    ds = AORCDataset(CFG, split="val")
    model = build_model().to(device)
    ckpt = torch.load(Path(CFG.output_dir) / "last.pt", map_location=device)
    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    edm = EDM(sigma_data=1.0)
    rng = np.random.default_rng(CFG.seed)
    sample_ids = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    samples = []
    for sample_num, idx in enumerate(sample_ids):
        cond, target = ds[int(idx)]
        zarr_idx, t_idx, i, j = ds.index[int(idx)]
        half = CFG.patch_size // 2
        lat = ds.lat[i - half:i + half]
        lon = ds.lon[j - half:j + half]

        pred = edm.sample(
            model,
            cond.unsqueeze(0).to(device),
            n_steps=CFG.sample_steps,
            n_target_channels=CFG.target_channels,
            device=str(device),
        )[0, 0].detach().cpu().numpy()

        truth_k = denorm_temp(target[0].numpy(), stats)
        coarse_k = denorm_temp(cond[3].numpy(), stats)
        pred_k = denorm_temp(pred, stats)

        samples.append(
            {
                "sample": sample_num,
                "t_idx": t_idx,
                "time": str(ds.zarr_times[zarr_idx][t_idx]),
                "lat": lat,
                "lon": lon,
                "truth_anom": anomaly(truth_k),
                "pred_anom": anomaly(pred_k),
                "coarse_anom": anomaly(coarse_k),
                "residual": pred_k - truth_k,
            }
        )
    return samples


def main():
    out_dir = Path(CFG.plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples()
    fig, axes = plt.subplots(len(samples), 4, figsize=(18, 4.2 * len(samples)), squeeze=False)

    for r, sample in enumerate(samples):
        extent = [
            float(sample["lon"][0]),
            float(sample["lon"][-1]),
            float(sample["lat"][0]),
            float(sample["lat"][-1]),
        ]
        fields = [
            ("AORC T anomaly", sample["truth_anom"], (-5, 5), "K"),
            ("Diffusion T anomaly", sample["pred_anom"], (-5, 5), "K"),
            ("Coarse AORC T anomaly", sample["coarse_anom"], (-5, 5), "K"),
            ("Diffusion - AORC", sample["residual"], (-1, 1), "K"),
        ]
        for c, (title, arr, lim, units) in enumerate(fields):
            ax = axes[r, c]
            im = ax.imshow(
                arr,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=lim[0],
                vmax=lim[1],
                interpolation="nearest",
            )
            ax.set_title(f"Sample {sample['sample']} | {title} | t={sample['t_idx']}", fontsize=13)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(alpha=0.18)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(units)

    fig.tight_layout()
    out = out_dir / "windstyle_temperature_anomaly_aorc_diffusion.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
