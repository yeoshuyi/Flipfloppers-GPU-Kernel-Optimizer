#!/bin/bash
#SBATCH --job-name=g2_4b_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -euo pipefail

apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    python3 /work/experiments/g2_4b_causal_compile_probe.py
