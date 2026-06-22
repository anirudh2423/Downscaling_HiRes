"""
Compute normalization statistics for coarse-AORC-conditioned temperature training.

This samples AORC 2 m temperature slabs rather than scanning the full year, and
interpolates ETOPO once onto the AORC grid for the static topography stats.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from config import CFG


N_AORC_SAMPLE_PER_ZARR = 25


def compute_aorc_temp_stats():
    print("Computing AORC TMP_2maboveground stats...", flush=True)
    rng = np.random.default_rng(CFG.seed)
    count = 0
    sum_x = 0.0
    sum_x2 = 0.0

    for path in CFG.aorc_zarr:
        ds = xr.open_zarr(
            path,
            consolidated=False,
            drop_variables=[
                "APCP_surface",
                "DLWRF_surface",
                "DSWRF_surface",
                "PRES_surface",
                "SPFH_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
            ],
        )
        t_count = ds.sizes["time"]
        t_sample = rng.choice(
            t_count,
            size=min(N_AORC_SAMPLE_PER_ZARR, t_count),
            replace=False,
        )
        for t in t_sample:
            slab = ds["TMP_2maboveground"].isel(time=int(t)).values
            valid = slab[~np.isnan(slab)].astype(np.float64)
            count += valid.size
            sum_x += valid.sum(dtype=np.float64)
            sum_x2 += np.square(valid, dtype=np.float64).sum(dtype=np.float64)
        ds.close()

    mean = sum_x / count
    var = sum_x2 / count - mean**2
    std = np.sqrt(max(var, 0.0))
    print(f"  AORC temp mean={mean:.6f} std={std:.6f}", flush=True)
    return float(mean), float(std)


def compute_topo_stats():
    print("Computing topo stats on AORC grid...", flush=True)
    ref_ds = xr.open_zarr(
        CFG.aorc_zarr[0],
        consolidated=False,
        drop_variables=[
            "APCP_surface",
            "DLWRF_surface",
            "DSWRF_surface",
            "PRES_surface",
            "SPFH_2maboveground",
            "TMP_2maboveground",
            "UGRD_10maboveground",
            "VGRD_10maboveground",
        ],
    )
    topo_ds = xr.open_dataset(CFG.topo_nc)
    topo_fine = (
        topo_ds["z"]
        .rename({"lat": "latitude", "lon": "longitude"})
        .interp(latitude=ref_ds.latitude, longitude=ref_ds.longitude, method="linear")
        .values.astype(np.float32)
    )
    ref_ds.close()
    topo_ds.close()

    valid = topo_fine[~np.isnan(topo_fine)]
    mean = float(valid.mean())
    std = float(valid.std())
    print(f"  Topo mean={mean:.6f} std={std:.6f}", flush=True)
    return mean, std


def main():
    out = Path(CFG.stats_file)
    if out.exists():
        print(f"Stats already exist at {out}, skipping.", flush=True)
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    temp_mean, temp_std = compute_aorc_temp_stats()
    topo_mean, topo_std = compute_topo_stats()

    np.savez(
        out,
        temp_mean=temp_mean,
        temp_std=temp_std,
        topo_mean=topo_mean,
        topo_std=topo_std,
    )
    print(f"Saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
