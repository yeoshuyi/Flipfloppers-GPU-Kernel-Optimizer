#!/bin/bash
#SBATCH --job-name=g4_6_ph1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:45:00
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

      echo "########## PHASE 0 (gate) ##########"
      python3 /work/experiments/g4_6_cutlass_stage0.py
      P0=$?
      echo "phase0 exit: $P0"
      if [ "$P0" -ne 0 ]; then
        echo "PHASE 0 GATE NOT MET -- Phase 1 deliberately NOT run."
        exit 0
      fi

      echo
      echo "########## PHASE 1 (raw throughput at the real shapes) ##########"
      python3 /work/experiments/g4_6_cutlass_phase1.py
      echo "phase1 exit: $?"
    '
