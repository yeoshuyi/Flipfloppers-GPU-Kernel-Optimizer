#!/bin/bash
# G8.0 -- evaluate Multi-head Latent Attention (and GQA/MQA) for the official
# matrix. MLA shrinks a DECODE-time KV cache; this harness only ever runs
# prefill, so the question is whether the absorbed form can still pay. Measures
# the real weight spectrum, accuracy vs latent rank on every scorable shape, a
# working absorbed-MLA A/B against the shipped path, and the row-14 cache.
# Touches nothing shipped.
#SBATCH --job-name=g8_0_mla_evaluation
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
    python3 /work/experiments/g8_0_mla_evaluation.py

J="${SLURM_JOB_ID}"
cp "/scratch/techjam2/runs/${J}.out" \
   "/scratch/work/results/logs/g8_0_mla_evaluation_run${J}.log" 2>/dev/null || true
