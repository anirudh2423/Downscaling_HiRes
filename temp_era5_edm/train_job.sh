#!/bin/bash
#PBS -N temp_era5_edm
#PBS -q workq
#PBS -l ncpus=8
#PBS -l mem=128gb
#PBS -l host=gpu-h100
#PBS -l gpu1=1
#PBS -l walltime=08:00:00
#PBS -o /Datastorage/scdlds_anirudhavireddy/TEMP_AORC/runs/temp_era5_edm/pbs_driver.log
#PBS -e /Datastorage/scdlds_anirudhavireddy/TEMP_AORC/runs/temp_era5_edm/pbs_driver.err
#PBS -j oe

set -o pipefail

PROJECT_DIR=/Datastorage/scdlds_anirudhavireddy/TEMP_AORC
LOG_DIR="$PROJECT_DIR/runs/temp_era5_edm"
LIVE_LOG="$LOG_DIR/train.log"

mkdir -p "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$LIVE_LOG") 2>&1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "PBS Job ID: $PBS_JOBID"

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES=0

/home/scdlds_anirudhavireddy/.conda/envs/diffusion/bin/python -u main.py

echo "Job finished: $(date)"
