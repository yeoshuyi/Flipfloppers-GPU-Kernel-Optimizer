#!/bin/bash
#SBATCH --job-name=g6_4a_v2_nb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    bash -c '
      echo "########## CONTROL (patch installed but disabled) ##########"
      python3 /work/experiments/g6_4a_v2_fp16_ffnin_newbudget.py --mode control
      echo "control exit: $?"
      echo
      echo "########## ffn_in FP16, new budget atol=0.002 rtol=0.02, 40 trials ##########"
      python3 /work/experiments/g6_4a_v2_fp16_ffnin_newbudget.py --mode fp16
      echo "fp16 exit: $?"
    '
