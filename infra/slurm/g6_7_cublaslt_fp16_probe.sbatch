#!/bin/bash
#SBATCH --job-name=g6_7_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

mkdir -p /scratch/work/.ext_build

apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    bash -c '
      # cpp_extension.load() needs a writable, container-visible TMPDIR --
      # the container default is not usable here (docs/PROGRESS.md step 32).
      export TMPDIR=/work/.ext_build
      export TORCH_EXT_BUILD_DIR=/work/.ext_build
      mkdir -p "$TMPDIR"
      python3 /work/probes/g6_7_cublaslt_fp16_probe.py
    '
