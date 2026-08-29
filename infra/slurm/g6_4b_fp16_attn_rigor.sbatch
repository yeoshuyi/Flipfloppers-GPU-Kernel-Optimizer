#!/bin/bash
#SBATCH --job-name=g6_4b_rigor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

run() {
    local name="$1"; shift
    echo "=== SHAPE: $name ==="
    apptainer exec --nv --cleanenv \
        --bind /scratch/work:/work \
        /scratch/kernel.sif \
        python3 /work/torch_transformer_benchmark.py --accuracy-trials 40 --benchmark-on-failure "$@" || true
    echo
}

run tiny               --batch-size 1   --seq-len 64
run default             --batch-size 8   --seq-len 128
run long_seq            --batch-size 8   --seq-len 1024
run large_batch         --batch-size 256 --seq-len 128
run default_padded      --batch-size 8   --seq-len 128  --padding-ratio 0.3
run long_seq_padded     --batch-size 8   --seq-len 1024 --padding-ratio 0.3
