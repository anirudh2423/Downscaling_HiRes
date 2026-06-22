# Downscaling_HiRes

Diffusion-based high-resolution weather downscaling experiments over CONUS.

This repository contains two training pipelines built around an EDM
(Elucidating Diffusion Models) denoising objective with a `diffusers.UNet2DModel`
backbone.

## Pipelines

### `wind_edm/`

High-resolution AORC 10 m wind downscaling.

- Target: `UGRD_10maboveground`, `VGRD_10maboveground`
- Patch size: `256 x 256`
- Conditioning:
  - topography
  - sky-view factor
  - cosine solar zenith angle
  - coarse AORC U wind
  - coarse AORC V wind
- Coarse wind construction:
  - normalize high-resolution AORC U/V patch
  - block-average from `256 x 256` to `32 x 32`
  - nearest-neighbor upsample back to `256 x 256`
  - feed as large-scale dynamic conditioning

### `temp_era5_edm/`

ERA5-conditioned high-resolution AORC 2 m temperature downscaling.

- Target: `TMP_2maboveground`
- Patch size: `256 x 256`
- Conditioning:
  - topography
  - sky-view factor
  - cosine solar zenith angle
  - ERA5 `2m_temperature`
- ERA5 source used on the HPC:
  - `/Datastorage/mihir.more/ALL_EXPERT_FORECASTS/era5_ground_truth_usa/arco_era5_2019_2026_usa_2m_temperature_and_total_precipitation_int16.zarr`
- AORC source used on the HPC:
  - `/Datastorage/scdlds_anirudhavireddy/aorc_2020_jan_jun_3hr.zarr`
  - `/Datastorage/scdlds_anirudhavireddy/aorc_2020_jul_dec_3hr.zarr`

### `temp_aorc_coarse_edm/`

Coarse-AORC-conditioned high-resolution AORC 2 m temperature refinement.

- Target: `TMP_2maboveground`
- Patch size: `256 x 256`
- Conditioning:
  - topography
  - sky-view factor
  - cosine solar zenith angle
  - coarse AORC `TMP_2maboveground`
- Coarse temperature construction:
  - normalize high-resolution AORC temperature patch
  - block-average from `256 x 256` to `32 x 32`
  - nearest-neighbor upsample back to `256 x 256`
  - feed as large-scale dynamic conditioning
- This experiment is closer to super-resolution/refinement because the coarse
  conditioning is derived from the same AORC field as the target.

## What is tracked

Tracked:

- model/training code
- dataset code
- stat/precompute scripts
- plotting/evaluation scripts
- PBS job scripts

Ignored:

- raw data
- zarr/netCDF files
- checkpoints
- logs
- TensorBoard event files
- generated plots
- Python cache files

## Typical Workflow

Create the Python environment first:

```bash
conda env create -f environment.yml
conda activate downscaling-hires
```

Or install into an existing Python environment with pip:

```bash
pip install -r requirements.txt
```

For either pipeline:

```bash
cd wind_edm
# or
cd temp_era5_edm
```

Compute statistics:

```bash
python -u compute_stats.py
# or for temperature
python -u compute_stats_temp.py
```

Run training locally:

```bash
python -u main.py
```

Submit on PBS:

```bash
qsub train_job.sh
```

Monitor:

```bash
tail -f runs/<run_name>/train.log
watch -n2 nvidia-smi
tensorboard --logdir runs/<run_name>/tb_logs --port 6006
```

Generate plots after training:

```bash
python -u make_comparison_plots.py
python -u make_us_map_plots.py
```

For the ERA5 temperature run:

```bash
python -u make_temperature_plots.py
```

For the coarse-AORC temperature run:

```bash
python -u make_temperature_plots.py
python -u make_temperature_windstyle_plot.py
```

## Notes

The code currently contains absolute HPC paths in `config.py`. Update those
paths before running on a different machine or filesystem.

The temperature experiment is EDM-trained. LCM-style sampling would require a
proper latent/consistency distillation setup or a compatible scheduler/model
pair; it is not silently used as a drop-in replacement for EDM training.
