#!/bin/bash
# G7.1 -- (a) calibrate the compiled causal path's activation footprint in
# BYTES so _would_oom_causal can be restated against real device capacity;
# (b) capture the Row-14 golden fingerprint that pins the current iteration as
# the benchmark (the frozen harness cannot score row 14).
#SBATCH --job-name=g7_1_gate_calibration
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
    python3 /work/experiments/g7_1_gate_calibration.py

J="${SLURM_JOB_ID}"
cp "/scratch/techjam2/runs/${J}.out" \
   "/scratch/work/results/logs/g7_1_gate_calibration_run${J}.log" 2>/dev/null || true
