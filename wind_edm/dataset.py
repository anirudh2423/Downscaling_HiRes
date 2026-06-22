import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from pathlib import Path
from datetime import datetime


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


def compute_czsa(lat, lon, time):
    doy      = time.timetuple().tm_yday
    hour_utc = time.hour + time.minute / 60.0
    decl     = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81))))
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    hour_angle = np.radians(15 * (hour_utc + lon_grid / 15.0 - 12))
    lat_r  = np.radians(lat_grid)
    csza   = (np.sin(lat_r) * np.sin(decl) +
              np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle))
    return np.clip(csza, 0, 1).astype(np.float32)


def make_coarse_condition(target, coarse_size):
    """
    Area-downsample normalized U/V to coarse_size, then nearest-upsample back
    to patch size. This preserves the large-scale wind state while removing
    the fine-scale detail the diffusion model should learn to generate.
    """
    c, h, w = target.shape
    if h % coarse_size != 0 or w % coarse_size != 0:
        raise ValueError(f"patch {(h, w)} must be divisible by coarse_size={coarse_size}")

    fy = h // coarse_size
    fx = w // coarse_size
    coarse = target.reshape(c, coarse_size, fy, coarse_size, fx).mean(axis=(2, 4))
    return np.repeat(np.repeat(coarse, fy, axis=1), fx, axis=2).astype(np.float32)


def _open_zarr(path, target_vars):
    """Open a single zarr, dropping all variables except target_vars."""
    keep           = set(target_vars)
    drop_variables = [v for v in AORC_VARS if v not in keep]
    return xr.open_zarr(path, consolidated=False, drop_variables=drop_variables)


def build_valid_index(zarr_paths, target_vars, patch_size, nan_threshold=0.6,
                      stride=128, seed=42):
    """
    Returns list of (zarr_idx, t_in_zarr, i, j).

    Fast approach: NaN coverage (land vs ocean) is static across timesteps,
    so we scan spatial positions from ONE reference slab per zarr, then
    pair every valid (i, j) with all timesteps. This avoids reading T full
    slabs per zarr (which is ~820 GB of I/O with the current chunk layout).
    """
    rng   = np.random.default_rng(seed)
    half  = patch_size // 2
    valid = []

    for zarr_idx, path in enumerate(zarr_paths):
        print(f"  Building spatial mask from zarr {zarr_idx}: {path}")
        ds = _open_zarr(path, target_vars)
        T  = ds.sizes["time"]
        H  = ds.sizes["latitude"]
        W  = ds.sizes["longitude"]

        # One mid-point reference slab to find land patch centres
        ref_t = T // 2
        ref_slabs = [ds[v][ref_t].values for v in target_vars]

        valid_centers = []
        for i in range(half, H - half, stride):
            for j in range(half, W - half, stride):
                patches = [s[i - half:i + half, j - half:j + half] for s in ref_slabs]
                if all(np.isnan(p).mean() < nan_threshold for p in patches):
                    valid_centers.append((i, j))

        print(f"    Valid spatial centres: {len(valid_centers)}  ×  {T} timesteps")

        # Every valid centre × every timestep
        for t in range(T):
            for i, j in valid_centers:
                valid.append((zarr_idx, t, i, j))

        ds.close()

    rng.shuffle(valid)
    return valid


class AORCDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg        = cfg
        self.patch_size = cfg.patch_size
        self.zarr_paths = cfg.aorc_zarr
        self.target_vars = cfg.target_vars
        self._ds_list   = [None] * len(self.zarr_paths)   # lazy per worker

        # Get grid coords from first zarr (lat/lon are identical across all zarrs)
        ds0       = _open_zarr(self.zarr_paths[0], self.target_vars)
        self.lat  = ds0.latitude.values
        self.lon  = ds0.longitude.values
        ds0.close()

        # Collect per-zarr time arrays (no concat into memory)
        self.zarr_times = []
        for path in self.zarr_paths:
            ds = _open_zarr(path, self.target_vars)
            self.zarr_times.append(ds.time.values)
            ds.close()
        # Flat combined time array for train/val split by time index
        self._all_times = np.concatenate(self.zarr_times)

        # topo — ETOPO uses 'z' var and lat/lon coords; rename to match AORC grid
        topo_ds   = xr.open_dataset(cfg.topo_nc)
        self.topo = (
            topo_ds["z"]
            .rename({"lat": "latitude", "lon": "longitude"})
            .interp(latitude=self.lat, longitude=self.lon, method="linear")
            .values.astype(np.float32)
        )
        topo_ds.close()

        # svf
        svf_ds   = xr.open_dataset(cfg.svf_nc)
        self.svf = svf_ds["svf"].values.astype(np.float32)
        svf_ds.close()

        # valid patch index (cached per output_dir)
        index_cache = Path(cfg.output_dir) / "valid_index.npy"
        if index_cache.exists():
            self.index = np.load(index_cache, allow_pickle=True).tolist()
        else:
            print("Building valid patch index (one-time, may take a while)...")
            self.index = build_valid_index(
                self.zarr_paths, self.target_vars,
                self.patch_size, cfg.nan_threshold,
            )
            index_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(index_cache, self.index)
        print(f"Total valid patches: {len(self.index)}")

        # train / val split by global time index
        # Build a mapping (zarr_idx, t_in_zarr) → global_t so we can split by time
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
            (z, t, i, j) for z, t, i, j in self.index
            if global_t(z, t) in t_set
        ]

        # subsample
        cap = cfg.max_patches if split == "train" else cfg.max_val_patches
        if cap and len(self.index) > cap:
            rng  = np.random.default_rng(cfg.seed)
            idxs = rng.choice(len(self.index), size=cap, replace=False)
            self.index = [self.index[k] for k in idxs]
        print(f"  {split} patches after subsample: {len(self.index)}")

        # normalisation stats
        stats = np.load(cfg.stats_file)
        self.ugrd_mean = float(stats["ugrd_mean"])
        self.ugrd_std  = float(stats["ugrd_std"])
        self.vgrd_mean = float(stats["vgrd_mean"])
        self.vgrd_std  = float(stats["vgrd_std"])
        self.topo_mean = float(stats["topo_mean"])
        self.topo_std  = float(stats["topo_std"])

    def _get_ds(self, zarr_idx):
        """Lazily open zarr per worker — avoids forking open file handles."""
        if self._ds_list[zarr_idx] is None:
            print(f"OPENING ZARR {zarr_idx} IN WORKER")
            self._ds_list[zarr_idx] = _open_zarr(
                self.zarr_paths[zarr_idx], self.target_vars
            )
        return self._ds_list[zarr_idx]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        zarr_idx, t_idx, i, j = self.index[idx]
        half   = self.patch_size // 2
        lat_sl = slice(i - half, i + half)
        lon_sl = slice(j - half, j + half)
        sl     = (lat_sl, lon_sl)

        ds = self._get_ds(zarr_idx)

        ugrd = ds["UGRD_10maboveground"].isel(
            time=t_idx, latitude=lat_sl, longitude=lon_sl
        ).values.astype(np.float32)
        vgrd = ds["VGRD_10maboveground"].isel(
            time=t_idx, latitude=lat_sl, longitude=lon_sl
        ).values.astype(np.float32)

        ugrd = np.nan_to_num((ugrd - self.ugrd_mean) / self.ugrd_std, nan=0.0)
        vgrd = np.nan_to_num((vgrd - self.vgrd_mean) / self.vgrd_std, nan=0.0)

        topo = (self.topo[sl] - self.topo_mean) / self.topo_std
        svf  = self.svf[sl]

        lat = self.lat[i - half:i + half]
        lon = self.lon[j - half:j + half]
        ts  = self.zarr_times[zarr_idx][t_idx].astype("datetime64[s]").astype(datetime)
        csza = compute_czsa(lat, lon, ts)

        target = np.stack([ugrd, vgrd],       axis=0)   # (2, H, W)
        coarse_uv = make_coarse_condition(target, self.cfg.coarse_size)

        cond = np.concatenate(
            [np.stack([topo, svf, csza], axis=0), coarse_uv],
            axis=0,
        )  # (5, H, W): topo, svf, csza, coarse U, coarse V

        return (
            torch.from_numpy(np.ascontiguousarray(cond)),
            torch.from_numpy(np.ascontiguousarray(target)),
        )
