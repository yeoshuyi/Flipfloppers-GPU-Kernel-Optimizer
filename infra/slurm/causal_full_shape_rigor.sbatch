#!/bin/bash
#SBATCH --job-name=causal_shapes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

run() {
    local name="$1"; shift
    echo "=== SHAPE: $name ==="
    apptainer exec --nv --cleanenv \
        --bind /scratch/work:/work \
        /scratch/kernel.sif \
        python3 /work/torch_transformer_benchmark.py --causal --accuracy-trials 40 --benchmark-on-failure "$@" || true
    echo
}

# Shapes not yet exercised on the causal path this session -- everything so
# far only tested causal at B=8,S=128 (default_causal/causal_padded).
run tiny_causal          --batch-size 1   --seq-len 64
run long_seq_causal      --batch-size 8   --seq-len 1024
run large_batch_causal   --batch-size 256 --seq-len 128
run tiny_causal_padded   --batch-size 1   --seq-len 64   --padding-ratio 0.3
run long_seq_causal_padded --batch-size 8 --seq-len 1024 --padding-ratio 0.3
run large_batch_causal_padded --batch-size 256 --seq-len 128 --padding-ratio 0.3
