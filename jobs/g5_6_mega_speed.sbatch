#!/bin/bash
#SBATCH --job-name=g5_6_spd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/techjam2/runs/%j.out
set -uo pipefail
mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build
apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif bash -c '
  python3 /work/probes/g5_5_mega_correct.py
  echo "======================== SPEED ========================"
  python3 /work/probes/g5_6_mega_speed.py
'
