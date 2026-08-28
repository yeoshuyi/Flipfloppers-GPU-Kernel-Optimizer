#!/bin/bash
#SBATCH --job-name=fp8_attn_precheck
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    python3 /work/experiments/fp8_attn_precheck.py --trials 20
