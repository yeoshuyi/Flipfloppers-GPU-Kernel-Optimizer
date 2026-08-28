#!/bin/bash
#SBATCH --job-name=g6_6_rigor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    python3 /work/benchmark.py --batch-size 1 --seq-len 64 --accuracy-trials 40 --benchmark-on-failure
