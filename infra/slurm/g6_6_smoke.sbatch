#!/bin/bash
#SBATCH --job-name=g6_6_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

mkdir -p /scratch/work/.ext_build

run() {
    local name="$1"; shift
    echo "=== SHAPE: $name ==="
    apptainer exec --nv --cleanenv \
        --bind /scratch/work:/work \
        /scratch/kernel.sif \
        bash -c "
          export TMPDIR=/work/.ext_build
          export TORCH_EXT_BUILD_DIR=/work/.ext_build
          mkdir -p \"\$TMPDIR\"
          python3 /work/torch_transformer_benchmark.py $* " || true
    echo
}

# Smoke only: the two shapes the G6.6 gate is supposed to fire on (tiny,
# default) plus one it must NOT fire on (long_seq, tok=8192 > _LT_MAX_TOKENS),
# so a silent gate inversion shows up before spending a full sweep.
run tiny      --batch-size 1 --seq-len 64
run default   --batch-size 8 --seq-len 128
run long_seq  --batch-size 8 --seq-len 1024
