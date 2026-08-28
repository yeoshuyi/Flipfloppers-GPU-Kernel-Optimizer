#!/bin/bash
#SBATCH --job-name=g6_0_capability
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -euo pipefail

apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    python3 /work/experiments/g6_0_capability_check.py
