#!/bin/bash
# G8.2 Phase 0 -- can FP16-accumulate + an FP32 carry fit the accuracy budget
# at the OFFICIAL matrix's K? The carry mechanism (SPLIT in g4_4_mma_gemm.cu)
# was measured once at K=512 and closed; the official rows run K=d_model=128 on
# ten of fourteen. No new CUDA -- this is the cheap gate before any CUTLASS work.
#SBATCH --job-name=g8_2_fp16accum_carry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=01:00:00
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
    python3 /work/experiments/g8_2_fp16accum_carry.py

J="${SLURM_JOB_ID}"
cp "/scratch/techjam2/runs/${J}.out" \
   "/scratch/work/results/logs/g8_2_fp16accum_carry_run${J}.log" 2>/dev/null || true
