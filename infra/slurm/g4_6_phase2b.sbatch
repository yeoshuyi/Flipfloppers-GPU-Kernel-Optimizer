#!/bin/bash
#SBATCH --job-name=g4_6_p2b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail
mkdir -p /scratch/work/.ext_build

apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    bash -c '
      export TMPDIR=/work/.ext_build
      export TORCH_EXT_BUILD_DIR=/work/.ext_build
      mkdir -p "$TMPDIR"
      echo "########## CONTROL (patch installed but disabled) ##########"
      python3 /work/probes/g4_6_cutlass_phase2b.py --mode control
      echo "control exit: $?"
      echo
      echo "########## CUTLASS FP16-ACCUMULATE ##########"
      python3 /work/probes/g4_6_cutlass_phase2b.py --mode cutlass
      echo "cutlass exit: $?"
    '
