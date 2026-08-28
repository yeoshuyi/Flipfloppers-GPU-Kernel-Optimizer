#!/bin/bash
#SBATCH --job-name=stage_bd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/techjam2/runs/%j.out
apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
    python3 /work/probes/stage_breakdown.py
