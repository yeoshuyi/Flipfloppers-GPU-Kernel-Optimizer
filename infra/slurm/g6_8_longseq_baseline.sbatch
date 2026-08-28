#!/bin/bash
#SBATCH --job-name=g6_8_ls_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

mkdir -p /scratch/work/.ext_build

# Fresh same-session long_seq baseline on the CURRENT shipped code, per step
# 33's environmental-drift lesson: never trust a sub-10% delta against a log
# from an earlier session.
apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    bash -c '
      export TMPDIR=/work/.ext_build
      export TORCH_EXT_BUILD_DIR=/work/.ext_build
      mkdir -p "$TMPDIR"
      python3 /work/benchmark.py --batch-size 8 --seq-len 1024
    ' || true
