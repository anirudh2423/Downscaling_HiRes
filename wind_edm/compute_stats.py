"""
compute_stats.py — run once before training.
Computes mean/std for UGRD, VGRD, and topo over the full 12-month dataset.
Saves norm_stats_wind.npz.

Processes each zarr independently to avoid loading the full concatenated
dataset into memory.
"""

import numpy as np
import xarray as xr
from pathlib import Path

AORC_ZARRS = [
    "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jan_jun_3hr.zarr",
    "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jul_dec_3hr.zarr",
]
TOPO_NC   = "/Datastorage/scdlds_anirudhavireddy/diffusion/us_topography_diffusion/data/topography/ETOPO_2022_v1_60s_N90W180_bed.nc"
OUT_STATS = "norm_stats_wind.npz"
N_SAMPLE_PER_ZARR = 25   # 25 × 2 zarrs = 50 total timesteps sampled


def compute_mean_std(zarr_paths, var_name):
    """Iterates each zarr separately to avoid in-memory concatenation."""
    print(f"Computing {var_name} stats ({N_SAMPLE_PER_ZARR} timesteps per zarr)...")
    count  = 0
    sum_x  = 0.0
    sum_x2 = 0.0
    rng = np.random.default_rng(42)

    for path in zarr_paths:
        ds = xr.open_zarr(path, consolidated=False)
        T  = len(ds.time)
        t_sample = rng.choice(T, size=min(N_SAMPLE_PER_ZARR, T), replace=False)
        for t in t_sample:
            slab  = ds[var_name][int(t)].values
            valid = slab[~np.isnan(slab)].astype(np.float64)
            count  += valid.size
            sum_x  += valid.sum(dtype=np.float64)
            sum_x2 += np.square(valid, dtype=np.float64).sum(dtype=np.float64)
        ds.close()

    mean = sum_x / count
    var  = sum_x2 / count - mean ** 2
    std  = np.sqrt(max(var, 0.0))
    print(f"  {var_name}  mean={mean:.6f}  std={std:.6f}")
    return float(mean), float(std)


def main():
    if Path(OUT_STATS).exists():
        print(f"Stats already exist at {OUT_STATS}, skipping.")
        return

    ugrd_mean, ugrd_std = compute_mean_std(AORC_ZARRS, "UGRD_10maboveground")
    vgrd_mean, vgrd_std = compute_mean_std(AORC_ZARRS, "VGRD_10maboveground")

    # Topo: use first zarr for AORC lat/lon coords (same grid across both zarrs)
    print("Computing topo stats (interpolated to AORC grid)...")
    ref_ds    = xr.open_zarr(AORC_ZARRS[0], consolidated=False)
    topo_ds   = xr.open_dataset(TOPO_NC)
    topo_fine = (
        topo_ds["z"]
        .rename({"lat": "latitude", "lon": "longitude"})
        .interp(latitude=ref_ds.latitude, longitude=ref_ds.longitude, method="linear")
        .values.astype(np.float32)
    )
    ref_ds.close()
    valid     = topo_fine[~np.isnan(topo_fine)]
    topo_mean = float(valid.mean())
    topo_std  = float(valid.std())
    print(f"  Topo  mean={topo_mean:.2f}  std={topo_std:.2f}")

    np.savez(
        OUT_STATS,
        ugrd_mean=ugrd_mean, ugrd_std=ugrd_std,
        vgrd_mean=vgrd_mean, vgrd_std=vgrd_std,
        topo_mean=topo_mean, topo_std=topo_std,
    )
    print(f"Saved → {OUT_STATS}")


if __name__ == "__main__":
    main()
