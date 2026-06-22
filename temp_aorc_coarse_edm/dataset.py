from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
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


def make_coarse_condition(target, coarse_size):
    """
    Area-downsample normalized target to coarse_size x coarse_size, then
    nearest-upsample back to patch size. This preserves the large-scale
    temperature state and removes fine detail for the diffusion model to learn.
    """
    channels, height, width = target.shape
    if height % coarse_size != 0 or width % coarse_size != 0:
        raise ValueError(f"patch {(height, width)} must be divisible by coarse_size={coarse_size}")

    fy = height // coarse_size
    fx = width // coarse_size
    coarse = target.reshape(channels, coarse_size, fy, coarse_size, fx).mean(axis=(2, 4))
    return np.repeat(np.repeat(coarse, fy, axis=1), fx, axis=2).astype(np.float32)


def _open_aorc(path, target_vars):
    keep = set(target_vars)
    drop_variables = [v for v in AORC_VARS if v not in keep]
    return xr.open_zarr(path, consolidated=False, drop_variables=drop_variables)


def build_valid_index(
    zarr_paths,
    target_vars,
    patch_size,
    nan_threshold=0.6,
    stride=128,
    seed=42,
):
    rng = np.random.default_rng(seed)
    half = patch_size // 2
    valid = []

    for zarr_idx, path in enumerate(zarr_paths):
        print(f"  Building spatial mask from zarr {zarr_idx}: {path}", flush=True)
        ds = _open_aorc(path, target_vars)
        time_count = ds.sizes["time"]
        height = ds.sizes["latitude"]
        width = ds.sizes["longitude"]

        ref_t = time_count // 2
        ref_slabs = [ds[v].isel(time=ref_t).values for v in target_vars]

        valid_centers = []
        for i in range(half, height - half, stride):
            for j in range(half, width - half, stride):
                patches = [
                    slab[i - half:i + half, j - half:j + half]
                    for slab in ref_slabs
                ]
                if all(np.isnan(p).mean() < nan_threshold for p in patches):
                    valid_centers.append((i, j))

        print(f"    Valid spatial centres: {len(valid_centers)} x {time_count} timesteps", flush=True)
        for t in range(time_count):
            for i, j in valid_centers:
                valid.append((zarr_idx, t, i, j))

        ds.close()

    rng.shuffle(valid)
    return valid


class AORCTemperatureCoarseDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.zarr_paths = cfg.aorc_zarr
        self.target_vars = cfg.target_vars
        self._ds_list = [None] * len(self.zarr_paths)

        ds0 = _open_aorc(self.zarr_paths[0], self.target_vars)
        self.lat = ds0.latitude.values.astype(np.float64)
        self.lon = ds0.longitude.values.astype(np.float64)
        ds0.close()

        self.zarr_times = []
        for path in self.zarr_paths:
            ds = _open_aorc(path, self.target_vars)
            self.zarr_times.append(ds.time.values.astype("datetime64[s]"))
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
            print("Building valid patch index for coarse AORC temperature...", flush=True)
            self.index = build_valid_index(
                self.zarr_paths,
                self.target_vars,
                self.patch_size,
                cfg.nan_threshold,
                stride=cfg.index_stride,
                seed=cfg.seed,
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
        self.topo_mean = float(stats["topo_mean"])
        self.topo_std = float(stats["topo_std"])

    def _get_ds(self, zarr_idx):
        if self._ds_list[zarr_idx] is None:
            print(f"OPENING AORC ZARR {zarr_idx} IN WORKER", flush=True)
            self._ds_list[zarr_idx] = _open_aorc(
                self.zarr_paths[zarr_idx],
                self.target_vars,
            )
        return self._ds_list[zarr_idx]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        zarr_idx, t_idx, i, j = self.index[idx]
        half = self.patch_size // 2
        lat_sl = slice(i - half, i + half)
        lon_sl = slice(j - half, j + half)
        sl = (lat_sl, lon_sl)

        ds = self._get_ds(zarr_idx)
        temp = ds["TMP_2maboveground"].isel(
            time=t_idx, latitude=lat_sl, longitude=lon_sl
        ).values.astype(np.float32)
        temp = np.nan_to_num((temp - self.temp_mean) / self.temp_std, nan=0.0)
        target = temp[None, :, :]
        coarse_temp = make_coarse_condition(target, self.cfg.coarse_size)

        topo = (self.topo[sl] - self.topo_mean) / self.topo_std
        svf = self.svf[sl]
        patch_lat = self.lat[i - half:i + half]
        patch_lon = self.lon[j - half:j + half]
        ts = self.zarr_times[zarr_idx][t_idx].astype("datetime64[s]").astype(datetime)
        csza = compute_csza(patch_lat, patch_lon, ts)

        cond = np.concatenate(
            [np.stack([topo, svf, csza], axis=0), coarse_temp],
            axis=0,
        )

        return (
            torch.from_numpy(np.ascontiguousarray(cond.astype(np.float32))),
            torch.from_numpy(np.ascontiguousarray(target.astype(np.float32))),
        )


AORCDataset = AORCTemperatureCoarseDataset
