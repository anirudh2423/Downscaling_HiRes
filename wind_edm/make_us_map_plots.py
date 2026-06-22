from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr

from config import CFG


def wind_speed(arr):
    return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)


def finite_limits(*arrays, q=(1, 99)):
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    return np.percentile(vals, q)


def patch_extent(lat, lon, center_i, center_j, patch_size):
    half = patch_size // 2
    lat0 = float(lat[center_i - half])
    lat1 = float(lat[center_i + half - 1])
    lon0 = float(lon[center_j - half])
    lon1 = float(lon[center_j + half - 1])
    return lon0, lon1, lat0, lat1


def add_patch_rect(ax, extent, label, color):
    lon0, lon1, lat0, lat1 = extent
    rect = Rectangle(
        (lon0, lat0),
        lon1 - lon0,
        lat1 - lat0,
        fill=False,
        lw=2.2,
        ec=color,
    )
    ax.add_patch(rect)
    ax.text(
        (lon0 + lon1) / 2,
        (lat0 + lat1) / 2,
        label,
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color=color,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2},
    )


def style_lonlat_axis(ax, title):
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25, lw=0.7)


def save_sample_location_map(out_dir, lat, lon, metadata, patch_size):
    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=170)
    ax.set_xlim(float(lon.min()), float(lon.max()))
    ax.set_ylim(float(lat.min()), float(lat.max()))
    style_lonlat_axis(ax, "Validation Samples on AORC CONUS Grid")

    domain = Rectangle(
        (float(lon.min()), float(lat.min())),
        float(lon.max() - lon.min()),
        float(lat.max() - lat.min()),
        fill=False,
        lw=2.0,
        ec="black",
    )
    ax.add_patch(domain)
    colors = ["#d00000", "#0066cc", "#008000", "#7b2cbf"]
    for k, (_, _, i, j) in enumerate(metadata):
        extent = patch_extent(lat, lon, int(i), int(j), patch_size)
        add_patch_rect(ax, extent, f"{k}", colors[k % len(colors)])

    ax.text(
        0.01,
        0.02,
        "Map uses native AORC lon/lat coordinates. Rectangles are 256x256 validation patches.",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(out_dir / "us_map_validation_sample_locations.png")
    plt.close(fig)


def save_georeferenced_speed_maps(out_dir, lat, lon, metadata, truth, pred):
    true_speed = wind_speed(truth)
    pred_speed = wind_speed(pred)
    vmin, vmax = finite_limits(true_speed, pred_speed)
    err_abs = max(np.percentile(np.abs(pred_speed - true_speed), 98), 1e-3)

    fig, axes = plt.subplots(len(metadata), 3, figsize=(13, 14), dpi=160)
    if len(metadata) == 1:
        axes = axes[None, :]
    for k, (_, t_idx, i, j) in enumerate(metadata):
        extent = patch_extent(lat, lon, int(i), int(j), CFG.patch_size)
        panels = [
            (true_speed[k], "AORC speed", "turbo", vmin, vmax),
            (pred_speed[k], "Diffusion speed", "turbo", vmin, vmax),
            (pred_speed[k] - true_speed[k], "Generated - AORC", "coolwarm", -err_abs, err_abs),
        ]
        for ax, (arr, title, cmap, lo, hi) in zip(axes[k], panels):
            im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi)
            style_lonlat_axis(ax, f"Sample {k} | {title} | t={int(t_idx)}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="m/s")
    fig.tight_layout()
    fig.savefig(out_dir / "us_map_georeferenced_aorc_vs_diffusion_speed.png")
    plt.close(fig)


def save_georeferenced_uv_maps(out_dir, lat, lon, metadata, truth, pred):
    uv_min, uv_max = finite_limits(truth[:, 0], pred[:, 0], truth[:, 1], pred[:, 1], q=(2, 98))
    fig, axes = plt.subplots(len(metadata), 4, figsize=(16, 14), dpi=150)
    if len(metadata) == 1:
        axes = axes[None, :]
    for k, (_, t_idx, i, j) in enumerate(metadata):
        extent = patch_extent(lat, lon, int(i), int(j), CFG.patch_size)
        panels = [
            (truth[k, 0], "AORC U"),
            (pred[k, 0], "Diffusion U"),
            (truth[k, 1], "AORC V"),
            (pred[k, 1], "Diffusion V"),
        ]
        for ax, (arr, title) in zip(axes[k], panels):
            im = ax.imshow(arr, origin="lower", extent=extent, cmap="RdBu_r", vmin=uv_min, vmax=uv_max)
            style_lonlat_axis(ax, f"Sample {k} | {title} | t={int(t_idx)}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="m/s")
    fig.tight_layout()
    fig.savefig(out_dir / "us_map_georeferenced_u_v_components.png")
    plt.close(fig)


def save_georeferenced_denoising_maps(out_dir, lat, lon, metadata, truth, noisy, denoised):
    true_speed = wind_speed(truth)
    noisy_speed = wind_speed(noisy)
    denoised_speed = wind_speed(denoised)
    vmin, vmax = finite_limits(true_speed, noisy_speed, denoised_speed)
    fig, axes = plt.subplots(len(metadata), 3, figsize=(13, 14), dpi=160)
    if len(metadata) == 1:
        axes = axes[None, :]
    for k, (_, t_idx, i, j) in enumerate(metadata):
        extent = patch_extent(lat, lon, int(i), int(j), CFG.patch_size)
        panels = [
            (true_speed[k], "AORC speed"),
            (noisy_speed[k], "Noisy diffusion input"),
            (denoised_speed[k], "EDM denoised speed"),
        ]
        for ax, (arr, title) in zip(axes[k], panels):
            im = ax.imshow(arr, origin="lower", extent=extent, cmap="turbo", vmin=vmin, vmax=vmax)
            style_lonlat_axis(ax, f"Sample {k} | {title} | t={int(t_idx)}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="m/s")
    fig.tight_layout()
    fig.savefig(out_dir / "us_map_georeferenced_denoising_speed.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(Path(CFG.output_dir) / "plots"))
    parser.add_argument("--pred_path", default=str(Path(CFG.output_dir) / "plots" / "sample_predictions.npz"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pred_path = Path(args.pred_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Could not find sample predictions at {pred_path}")

    with xr.open_zarr(CFG.aorc_zarr[0], consolidated=False, drop_variables=[]) as ds:
        lat = ds.latitude.values
        lon = ds.longitude.values

    data = np.load(pred_path)
    metadata = data["metadata"]
    truth = data["truth"]
    pred = data["pred"]
    noisy = data["noisy"]
    denoised = data["denoised"]

    out_dir.mkdir(parents=True, exist_ok=True)
    save_sample_location_map(out_dir, lat, lon, metadata, CFG.patch_size)
    save_georeferenced_speed_maps(out_dir, lat, lon, metadata, truth, pred)
    save_georeferenced_uv_maps(out_dir, lat, lon, metadata, truth, pred)
    save_georeferenced_denoising_maps(out_dir, lat, lon, metadata, truth, noisy, denoised)
    print(f"Saved US map plots to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
