from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Data ---
    aorc_zarr: list = field(default_factory=lambda: [
        "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jan_jun_3hr.zarr",
        "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jul_dec_3hr.zarr",
    ])
    era5_zarr: str = (
        "/Datastorage/mihir.more/ALL_EXPERT_FORECASTS/"
        "era5_ground_truth_usa/"
        "arco_era5_2019_2026_usa_2m_temperature_and_total_precipitation_int16.zarr"
    )
    topo_nc: str = (
        "/Datastorage/scdlds_anirudhavireddy/diffusion/us_topography_diffusion/"
        "data/topography/ETOPO_2022_v1_60s_N90W180_bed.nc"
    )
    svf_nc: str = "/Datastorage/scdlds_anirudhavireddy/Scipts_Training/svf_conus.nc"
    stats_file: str = "/Datastorage/scdlds_anirudhavireddy/TEMP_AORC/norm_stats_temp_era5.npz"
    output_dir: str = "/Datastorage/scdlds_anirudhavireddy/TEMP_AORC/runs/temp_era5_edm"
    plots_dir: str = "/Datastorage/scdlds_anirudhavireddy/TEMP_AORC/Plots"

    target_vars: list = field(default_factory=lambda: ["TMP_2maboveground"])
    target_channels: int = 1
    patch_size: int = 256
    cond_channels: int = 4       # topo, SVF, CSZA, ERA5 t2m
    input_channels: int = 5      # 1 noisy temperature channel + cond_channels
    val_fraction: float = 0.1
    nan_threshold: float = 0.6
    index_stride: int = 128
    era5_cache_size: int = 8

    # Normalisation stats; compute_stats_temp.py overwrites these at runtime.
    temp_mean: float = 290.0
    temp_std: float = 12.0
    era5_t2m_mean: float = 290.0
    era5_t2m_std: float = 12.0
    topo_mean: float = -833.606
    topo_std: float = 2315.990

    # --- Diffusion / sampling ---
    sampler: str = "edm"         # LCM is inference-only unless distilled; keep EDM training.
    sample_steps: int = 18
    lcm_steps: int = 4           # reserved for distilled/LCM-compatible checkpoints.

    # --- Model ---
    base_ch: int = 64
    ch_mult: tuple = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attn_resolutions: tuple = (32, 16)

    # --- Training ---
    max_patches: int = 25000
    max_val_patches: int = 2000
    batch_size: int = 64
    grad_accum_steps: int = 1
    lr: float = 1e-4
    num_epochs: int = 20
    num_workers: int = 4
    prefetch_factor: int = 4
    grad_clip: float = 1.0
    save_every: int = 5
    val_every: int = 5
    log_every: int = 10
    seed: int = 42
    amp: bool = True
    amp_dtype: str = "bf16"
    device: str = "cuda"
    compile_model: bool = True

    # --- GPU monitoring ---
    gpu_util_low: int = 90
    gpu_util_high: int = 95


CFG = Config()
