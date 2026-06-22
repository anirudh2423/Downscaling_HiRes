"""
precompute.py
Saves:
  - svf_conus.nc : (H, W) float32  sky-view factor on the AORC grid
"""

import numpy as np
import xarray as xr
from pathlib import Path

AORC_ZARRS = [
    "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jan_jun_3hr.zarr",
    "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jul_dec_3hr.zarr",
]
TOPO_NC = "/Datastorage/scdlds_anirudhavireddy/diffusion/us_topography_diffusion/data/topography/ETOPO_2022_v1_60s_N90W180_bed.nc"
OUT_SVF = "svf_conus.nc"


def compute_svf(altitude, res_m=927.0):
    """SVF ≈ cos(slope),  altitude in metres, res_m = pixel size."""
    dz_dy = np.gradient(altitude, res_m, axis=0)
    dz_dx = np.gradient(altitude, res_m, axis=1)
    slope  = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    return np.cos(slope).astype(np.float32)


def load_topo_on_aorc_grid(topo_nc, lat, lon):
    """
    Reads ETOPO (var='z', coords 'lat'/'lon') and interpolates
    onto the AORC latitude/longitude grid.
    """
    topo_ds   = xr.open_dataset(topo_nc)
    topo_fine = (
        topo_ds["z"]
        .rename({"lat": "latitude", "lon": "longitude"})
        .interp(latitude=lat, longitude=lon, method="linear")
        .values.astype(np.float32)
    )
    return topo_fine


def main():
    if Path(OUT_SVF).exists():
        print(f"SVF already exists at {OUT_SVF}, skipping.")
        return

    print("Loading AORC coords for target grid...")
    ds  = xr.open_zarr(AORC_ZARRS[0], consolidated=False)
    lat = ds.latitude.values
    lon = ds.longitude.values

    print("Loading and interpolating topo to AORC grid...")
    topo_fine = load_topo_on_aorc_grid(TOPO_NC, lat, lon)
    print(f"  topo shape: {topo_fine.shape}  min={topo_fine.min():.1f}  max={topo_fine.max():.1f}")

    print("Computing SVF...")
    svf = compute_svf(topo_fine)

    print("Saving SVF...")
    xr.DataArray(
        svf,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        name="svf",
    ).to_netcdf(OUT_SVF)

    print(f"Done — saved SVF → {OUT_SVF}")


if __name__ == "__main__":
    main()
