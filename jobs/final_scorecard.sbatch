#!/bin/bash
#SBATCH --job-name=final_scorecard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail
mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build

apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
    python3 /work/probes/final_scorecard.py

J="${SLURM_JOB_ID}"
cp "/scratch/techjam2/runs/${J}.out" "/scratch/work/results/final_scorecard_run${J}.log" 2>/dev/null || true
