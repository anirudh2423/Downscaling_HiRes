from dataclasses import dataclass, field


@dataclass
class Config:
    aorc_zarr: list = field(default_factory=lambda: [
        "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jan_jun_3hr.zarr",
        "/Datastorage/scdlds_anirudhavireddy/aorc_2020_jul_dec_3hr.zarr",
    ])
    topo_nc: str = "/Datastorage/scdlds_anirudhavireddy/diffusion/us_topography_diffusion/data/topography/ETOPO_2022_v1_60s_N90W180_bed.nc"
    svf_nc:  str = "svf_conus.nc"
    stats_file: str = "norm_stats_wind.npz"
    output_dir: str = "runs/wind_edm_coarse"

    target_vars: list = field(default_factory=lambda: [
        "UGRD_10maboveground",
        "VGRD_10maboveground",
    ])
    patch_size:    int   = 256
    coarse_size:   int   = 32
    cond_channels: int   = 5       # topo, SVF, CSZA, coarse U, coarse V
    input_channels:int   = 7       # 2 noisy wind channels + cond_channels
    val_fraction:  float = 0.1
    nan_threshold: float = 0.6

    # norm stats — placeholders until compute_stats.py is run
    ugrd_mean: float = 0.0
    ugrd_std:  float = 1.0
    vgrd_mean: float = 0.0
    vgrd_std:  float = 1.0
    topo_mean: float = -833.606
    topo_std:  float = 2315.990

    # --- Diffusion ---
    T:          int   = 1000
    beta_start: float = 1e-4
    beta_end:   float = 0.02
    schedule:   str   = "cosine"

    # --- Model ---
    base_ch:          int   = 64
    ch_mult:          tuple = (1, 2, 4, 8)
    num_res_blocks:   int   = 2
    attn_resolutions: tuple = (32, 16)

    # --- Training ---
    max_patches:      int   = 25000
    max_val_patches:  int   = 2000
    batch_size:       int   = 64
    grad_accum_steps: int   = 1       # effective batch = batch_size * grad_accum_steps
    lr:               float = 1e-4
    num_epochs:       int   = 20
    num_workers:      int   = 4
    prefetch_factor:  int   = 4
    grad_clip:        float = 1.0
    save_every:       int   = 5
    val_every:        int   = 5
    log_every:        int   = 10
    seed:             int   = 42
    amp:              bool  = True
    amp_dtype:        str   = "bf16"
    device:           str   = "cuda"
    compile_model:    bool  = True

    # --- GPU monitoring ---
    gpu_util_low:  int = 90
    gpu_util_high: int = 95


CFG = Config()
