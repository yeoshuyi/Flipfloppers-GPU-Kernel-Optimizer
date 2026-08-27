#!/bin/bash
#SBATCH --job-name=g4_0_ceil
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

mkdir -p /scratch/work/.ext_build

run() {
    echo "############ $1 ############"
    apptainer exec --nv --cleanenv \
        --bind /scratch/work:/work \
        /scratch/kernel.sif \
        bash -c "
          export TMPDIR=/work/.ext_build
          export TORCH_EXT_BUILD_DIR=/work/.ext_build
          mkdir -p \"\$TMPDIR\"
          python3 /work/probes/$1
        " || true
    echo
}

run g4_0_ceiling.py

