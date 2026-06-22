from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import map_coordinates
from torch.utils.data import Dataset


AORC_VARS = (
    "APCP_surface",
    "DLWRF_surface",
    "DSWRF_surface",
    "PRES_surface",
    "SPFH_2maboveground",
    "TMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
)


def compute_csza(lat, lon, time):
    doy = time.timetuple().tm_yday
    hour_utc = time.hour + time.minute / 60.0
    decl = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81))))
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    hour_angle = np.radians(15 * (hour_utc + lon_grid / 15.0 - 12))
    lat_r = np.radians(lat_grid)
    csza = (
        np.sin(lat_r) * np.sin(decl)
        + np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle)
    )
    return np.clip(csza, 0, 1).astype(np.float32)


def _open_aorc(path, target_vars):
    keep = set(target_vars)
    drop_variables = [v for v in AORC_VARS if v not in keep]
    return xr.open_zarr(path, consolidated=False, drop_variables=drop_variables)


def _open_era5(path):
    return xr.open_zarr(path, consolidated=None)


def _as_seconds(values):
    return values.astype("datetime64[s]")


def build_valid_index(
    zarr_paths,
    target_vars,
    patch_size,
    nan_threshold=0.6,
    stride=128,
    seed=42,
    lat=None,
    lon=None,
    era5_lat_min=None,
    era5_lat_max=None,
    era5_lon_min_360=None,
    era5_lon_max_360=None,
):
    """
    Returns list of (zarr_idx, t_in_zarr, i, j).

    The spatial mask is built from one reference slab per zarr. Centers are
    also restricted to patch extents fully covered by the ERA5 USA domain.
    """
    rng = np.random.default_rng(seed)
    half = patch_size // 2
    valid = []

    for zarr_idx, path in enumerate(zarr_paths):
        print(f"  Building spatial mask from zarr {zarr_idx}: {path}", flush=True)
        ds = _open_aorc(path, target_vars)
        t_count = ds.sizes["time"]
        height = ds.sizes["latitude"]
        width = ds.sizes["longitude"]

        ref_t = t_count // 2
        ref_slabs = [ds[v][ref_t].values for v in target_vars]

        valid_centers = []
        for i in range(half, height - half, stride):
            patch_lat = lat[i - half:i + half]
            if era5_lat_min is not None:
                if patch_lat.min() < era5_lat_min or patch_lat.max() > era5_lat_max:
                    continue

            for j in range(half, width - half, stride):
                patch_lon_360 = np.mod(lon[j - half:j + half], 360.0)
                if era5_lon_min_360 is not None:
                    if (
                        patch_lon_360.min() < era5_lon_min_360
                        or patch_lon_360.max() > era5_lon_max_360
                    ):
                        continue

                patches = [
                    s[i - half:i + half, j - half:j + half] for s in ref_slabs
                ]
                if all(np.isnan(p).mean() < nan_threshold for p in patches):
                    valid_centers.append((i, j))

        print(f"    Valid spatial centres: {len(valid_centers)} x {t_count} timesteps", flush=True)

        for t in range(t_count):
            for i, j in valid_centers:
                valid.append((zarr_idx, t, i, j))

        ds.close()

    rng.shuffle(valid)
    return valid


class AORCTemperatureDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.zarr_paths = cfg.aorc_zarr
        self.target_vars = cfg.target_vars
        self._aorc_ds_list = [None] * len(self.zarr_paths)
        self._era5_ds = None
        self._era5_cache = OrderedDict()

        ds0 = _open_aorc(self.zarr_paths[0], self.target_vars)
        self.lat = ds0.latitude.values.astype(np.float64)
        self.lon = ds0.longitude.values.astype(np.float64)
        ds0.close()

        era5_meta = _open_era5(cfg.era5_zarr)
        self.era5_var = "2m_temperature"
        self.era5_lat = era5_meta.lat.values.astype(np.float64)
        self.era5_lon = era5_meta.lon.values.astype(np.float64)
        self.era5_time = _as_seconds(era5_meta.time.values)
        self.era5_lat_min = float(np.nanmin(self.era5_lat))
        self.era5_lat_max = float(np.nanmax(self.era5_lat))
        self.era5_lon_min = float(np.nanmin(self.era5_lon))
        self.era5_lon_max = float(np.nanmax(self.era5_lon))
        era5_meta.close()

        self.era5_time_to_idx = {t: k for k, t in enumerate(self.era5_time)}

        self.zarr_times = []
        for path in self.zarr_paths:
            ds = _open_aorc(path, self.target_vars)
            times = _as_seconds(ds.time.values)
            missing = [str(t) for t in times if t not in self.era5_time_to_idx]
            if missing:
                raise ValueError(
                    f"{len(missing)} AORC times from {path} are missing in ERA5; "
                    f"first missing: {missing[0]}"
                )
            self.zarr_times.append(times)
            ds.close()

        topo_ds = xr.open_dataset(cfg.topo_nc)
        self.topo = (
            topo_ds["z"]
            .rename({"lat": "latitude", "lon": "longitude"})
            .interp(latitude=self.lat, longitude=self.lon, method="linear")
            .values.astype(np.float32)
        )
        topo_ds.close()

        svf_ds = xr.open_dataset(cfg.svf_nc)
        self.svf = svf_ds["svf"].values.astype(np.float32)
        svf_ds.close()

        index_cache = Path(cfg.output_dir) / "valid_index.npy"
        if index_cache.exists():
            self.index = np.load(index_cache, allow_pickle=True).tolist()
        else:
            print("Building valid patch index for ERA5-covered AORC temperature...", flush=True)
            self.index = build_valid_index(
                self.zarr_paths,
                self.target_vars,
                self.patch_size,
                cfg.nan_threshold,
                stride=cfg.index_stride,
                seed=cfg.seed,
                lat=self.lat,
                lon=self.lon,
                era5_lat_min=self.era5_lat_min,
                era5_lat_max=self.era5_lat_max,
                era5_lon_min_360=self.era5_lon_min,
                era5_lon_max_360=self.era5_lon_max,
            )
            index_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(index_cache, self.index)
        print(f"Total valid patches: {len(self.index)}", flush=True)

        offsets = [0]
        for zt in self.zarr_times:
            offsets.append(offsets[-1] + len(zt))

        def global_t(zarr_idx, t_in_zarr):
            return offsets[zarr_idx] + t_in_zarr

        all_global_t = sorted(set(global_t(z, t) for z, t, _, _ in self.index))
        n_val = max(1, int(len(all_global_t) * cfg.val_fraction))
        t_set = (
            set(all_global_t[:-n_val]) if split == "train"
            else set(all_global_t[-n_val:])
        )
        self.index = [
            (z, t, i, j)
            for z, t, i, j in self.index
            if global_t(z, t) in t_set
        ]

        cap = cfg.max_patches if split == "train" else cfg.max_val_patches
        if cap and len(self.index) > cap:
            rng = np.random.default_rng(cfg.seed)
            idxs = rng.choice(len(self.index), size=cap, replace=False)
            self.index = [self.index[k] for k in idxs]
        print(f"  {split} patches after subsample: {len(self.index)}", flush=True)

        stats = np.load(cfg.stats_file)
        self.temp_mean = float(stats["temp_mean"])
        self.temp_std = float(stats["temp_std"])
        self.era5_t2m_mean = float(stats["era5_t2m_mean"])
        self.era5_t2m_std = float(stats["era5_t2m_std"])
        self.topo_mean = float(stats["topo_mean"])
        self.topo_std = float(stats["topo_std"])

    def _get_aorc_ds(self, zarr_idx):
        if self._aorc_ds_list[zarr_idx] is None:
            print(f"OPENING AORC ZARR {zarr_idx} IN WORKER", flush=True)
            self._aorc_ds_list[zarr_idx] = _open_aorc(
                self.zarr_paths[zarr_idx], self.target_vars
            )
        return self._aorc_ds_list[zarr_idx]

    def _get_era5_ds(self):
        if self._era5_ds is None:
            print("OPENING ERA5 ZARR IN WORKER", flush=True)
            self._era5_ds = _open_era5(self.cfg.era5_zarr)
        return self._era5_ds

    def _get_era5_slab(self, era5_time_idx):
        if era5_time_idx in self._era5_cache:
            slab = self._era5_cache.pop(era5_time_idx)
            self._era5_cache[era5_time_idx] = slab
            return slab

        ds = self._get_era5_ds()
        slab = ds[self.era5_var].isel(time=era5_time_idx).values.astype(np.float32)
        self._era5_cache[era5_time_idx] = slab
        while len(self._era5_cache) > self.cfg.era5_cache_size:
            self._era5_cache.popitem(last=False)
        return slab

    def _interp_era5_patch(self, era5_slab, patch_lat, patch_lon):
        lat0 = self.era5_lat[0]
        lon0 = self.era5_lon[0]
        dlat = abs(self.era5_lat[1] - self.era5_lat[0])
        dlon = abs(self.era5_lon[1] - self.era5_lon[0])

        lat_idx = (lat0 - patch_lat) / dlat
        lon_idx = (np.mod(patch_lon, 360.0) - lon0) / dlon
        yy, xx = np.meshgrid(lat_idx, lon_idx, indexing="ij")
        return map_coordinates(
            era5_slab,
            [yy, xx],
            order=1,
            mode="nearest",
        ).astype(np.float32)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        zarr_idx, t_idx, i, j = self.index[idx]
        half = self.patch_size // 2
        lat_sl = slice(i - half, i + half)
        lon_sl = slice(j - half, j + half)
        sl = (lat_sl, lon_sl)

        ds = self._get_aorc_ds(zarr_idx)
        temp = ds["TMP_2maboveground"].isel(
            time=t_idx, latitude=lat_sl, longitude=lon_sl
        ).values.astype(np.float32)
        temp = np.nan_to_num((temp - self.temp_mean) / self.temp_std, nan=0.0)

        patch_lat = self.lat[i - half:i + half]
        patch_lon = self.lon[j - half:j + half]
        timestamp = self.zarr_times[zarr_idx][t_idx]
        era5_idx = self.era5_time_to_idx[timestamp]
        era5_patch = self._interp_era5_patch(
            self._get_era5_slab(era5_idx),
            patch_lat,
            patch_lon,
        )
        era5_patch = (era5_patch - self.era5_t2m_mean) / self.era5_t2m_std

        topo = (self.topo[sl] - self.topo_mean) / self.topo_std
        svf = self.svf[sl]
        ts = timestamp.astype("datetime64[s]").astype(datetime)
        csza = compute_csza(patch_lat, patch_lon, ts)

        target = temp[None, :, :]
        cond = np.stack([topo, svf, csza, era5_patch], axis=0)

        return (
            torch.from_numpy(np.ascontiguousarray(cond.astype(np.float32))),
            torch.from_numpy(np.ascontiguousarray(target.astype(np.float32))),
        )


# Keep the training code import unchanged.
AORCDataset = AORCTemperatureDataset
