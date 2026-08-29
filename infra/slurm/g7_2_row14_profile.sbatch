#!/bin/bash
# G7.2 -- Step 0: profile _chunked_forward_causal at official Row 14 to
# decide where roadmap steps 1-7 actually pay off. No benchmark.py change.
#SBATCH --job-name=g7_2_row14_profile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=01:30:00
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
    python3 /work/experiments/g7_2_row14_profile.py

J="${SLURM_JOB_ID}"
cp "/scratch/techjam2/runs/${J}.out" \
   "/scratch/work/results/logs/g7_2_row14_profile_run${J}.log" 2>/dev/null || true
