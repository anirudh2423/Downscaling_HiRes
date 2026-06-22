#!/bin/bash
#PBS -N wind_edm_coarse
#PBS -q workq
#PBS -l ncpus=8
#PBS -l mem=96gb
#PBS -l host=gpu-h100
#PBS -l gpu1=1
#PBS -l walltime=06:00:00
#PBS -o /Datastorage/scdlds_anirudhavireddy/Scipts_Training/runs/wind_edm_coarse/pbs_driver.log
#PBS -e /Datastorage/scdlds_anirudhavireddy/Scipts_Training/runs/wind_edm_coarse/pbs_driver.err
#PBS -j oe

set -o pipefail

LOG_DIR=/Datastorage/scdlds_anirudhavireddy/Scipts_Training/runs/wind_edm_coarse
LIVE_LOG="$LOG_DIR/train.log"

mkdir -p "$LOG_DIR"

exec > >(stdbuf -oL tee -a "$LIVE_LOG") 2>&1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "PBS Job ID: $PBS_JOBID"

cd /Datastorage/scdlds_anirudhavireddy/Scipts_Training

export CUDA_VISIBLE_DEVICES=0

/home/scdlds_anirudhavireddy/.conda/envs/diffusion/bin/python -u main.py

echo "Job finished: $(date)"
