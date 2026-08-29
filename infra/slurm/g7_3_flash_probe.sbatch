#!/bin/bash
# G7.3 -- Steps 1+2: is flash faster than mem-efficient for Row 14 per-chunk
# attention, and is the raw flash op bottom-right aligned for Lq < Lk?
#SBATCH --job-name=g7_3_flash_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail
mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apptainer exec --nv --cleanenv --bind /scratch/work:/work \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env SLURM_JOB_ID="${SLURM_JOB_ID}" \
    /scratch/kernel.sif \
    python3 /work/experiments/g7_3_flash_probe.py

J="${SLURM_JOB_ID}"
cp "/scratch/techjam2/runs/${J}.out" \
   "/scratch/work/results/logs/g7_3_flash_probe_run${J}.log" 2>/dev/null || true
