#!/bin/bash
#PBS -N temp_aorc_coarse_plots
#PBS -q workq
#PBS -l ncpus=4
#PBS -l mem=64gb
#PBS -l host=gpu-h100
#PBS -l gpu1=1
#PBS -l walltime=02:00:00
#PBS -o /Datastorage/scdlds_anirudhavireddy/Temp_AORC_Coarse/Plots/pbs_plot.log
#PBS -e /Datastorage/scdlds_anirudhavireddy/Temp_AORC_Coarse/Plots/pbs_plot.err
#PBS -j oe

set -o pipefail

mkdir -p /Datastorage/scdlds_anirudhavireddy/Temp_AORC_Coarse/Plots

echo "Plot job started: $(date)"
echo "Node: $(hostname)"
echo "PBS Job ID: $PBS_JOBID"

cd /Datastorage/scdlds_anirudhavireddy/Temp_AORC_Coarse

export CUDA_VISIBLE_DEVICES=0

/home/scdlds_anirudhavireddy/.conda/envs/diffusion/bin/python -u make_temperature_plots.py

echo "Plot job finished: $(date)"
