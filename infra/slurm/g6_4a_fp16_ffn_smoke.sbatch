#!/bin/bash
#SBATCH --job-name=g6_4a_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -euo pipefail

apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    python3 /work/torch_transformer_benchmark.py --batch-size 1 --seq-len 64 --benchmark-on-failure
