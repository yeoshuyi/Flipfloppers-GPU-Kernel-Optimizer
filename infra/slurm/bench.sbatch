#!/bin/bash
#SBATCH --job-name=kbench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -euo pipefail
CANDIDATE="$1"; SHAPE_CFG="$2"; MODE="${3:-bench}"

apptainer exec --nv --cleanenv \
  --bind /scratch/work:/work --bind /scratch/techjam2/runs:/runs \
  /scratch/kernel.sif \
  python3 /work/harness/run.py \
      --candidate "$CANDIDATE" --shape "$SHAPE_CFG" --mode "$MODE" \
      --out "/runs/$SLURM_JOB_ID.json"
