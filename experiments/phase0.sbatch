#!/bin/bash
#SBATCH --job-name=phase0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/runs/%j.out

# TEMPORARY: runs the venv's python directly instead of through the Apptainer
# image described in docs/SETUP.md #3 -- /scratch/kernel.sif does not exist
# yet. Replace this with the apptainer exec form once the image is built.
set -euo pipefail
/home/techjam2/.venv/bin/python3 /scratch/work/experiments/phase0.py > "/scratch/runs/${SLURM_JOB_ID}.json"
