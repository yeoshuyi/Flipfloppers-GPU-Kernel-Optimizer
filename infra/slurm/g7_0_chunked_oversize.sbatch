#!/bin/bash
# G7.0 -- verify _chunked_forward_causal: the shipped model executes official
# Row 14 (B=32 d=1024 H=16 S=100000 L=2 ffn=1024, causal) and larger causal
# shapes on one 24 GB RTX 4090 via sequence chunking. The frozen harness cannot
# score Row 14 (FP32 reference OOMs first), so this is the standalone proof.
#SBATCH --job-name=g7_0_chunked_oversize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail
mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build
# tight ~20 GB Row-14 budget -- reduce allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apptainer exec --nv --cleanenv --bind /scratch/work:/work \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /scratch/kernel.sif \
    python3 /work/experiments/g7_0_chunked_oversize.py

J="${SLURM_JOB_ID}"
cp "/scratch/techjam2/runs/${J}.out" \
   "/scratch/work/results/logs/g7_0_chunked_oversize_run${J}.log" 2>/dev/null || true
