#!/bin/bash
#SBATCH --job-name=g6_9_lt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/techjam2/runs/%j.out
set -uo pipefail
mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build
apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
    python3 /work/probes/g6_9_lt_official_sweep.py
