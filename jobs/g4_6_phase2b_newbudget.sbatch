#!/bin/bash
#SBATCH --job-name=g4_6_p2b_nb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=01:20:00
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
      echo "########## CONTROL (patch installed but disabled), route=both ##########"
      python3 /work/probes/g4_6_cutlass_phase2b_newbudget.py --mode control --route both
      echo "control exit: $?"
      echo
      echo "########## CUTLASS FP16-ACCUMULATE, route=both, new budget atol=0.002 rtol=0.02, 40 trials ##########"
      python3 /work/probes/g4_6_cutlass_phase2b_newbudget.py --mode cutlass --route both
      echo "cutlass(both) exit: $?"
      echo
      echo "########## CUTLASS FP16-ACCUMULATE, route=qkv (closest near-miss at old budget) ##########"
      python3 /work/probes/g4_6_cutlass_phase2b_newbudget.py --mode cutlass --route qkv
      echo "cutlass(qkv) exit: $?"
    '
