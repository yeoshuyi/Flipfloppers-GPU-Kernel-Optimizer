#!/bin/bash
#SBATCH --job-name=causal4_ship
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
        python3 /work/benchmark.py --accuracy-trials 40 --benchmark-on-failure "$@" || true
    echo
}

# Full non-causal 8-shape sweep (dead fallback -- must stay unaffected)
run tiny               --batch-size 1   --seq-len 64
run default             --batch-size 8   --seq-len 128
run long_seq            --batch-size 8   --seq-len 1024
run large_batch         --batch-size 256 --seq-len 128
run default_padded      --batch-size 8   --seq-len 128  --padding-ratio 0.3
run long_seq_padded     --batch-size 8   --seq-len 1024 --padding-ratio 0.3

# All four causal shapes, per the plan's updated verification bar
run default_causal            --batch-size 8   --seq-len 128  --causal
run causal_padded             --batch-size 8   --seq-len 128  --causal --padding-ratio 0.3
run tiny_causal                --batch-size 1   --seq-len 64   --causal
run tiny_causal_padded         --batch-size 1   --seq-len 64   --causal --padding-ratio 0.3
run long_seq_causal            --batch-size 8   --seq-len 1024 --causal
run long_seq_causal_padded     --batch-size 8   --seq-len 1024 --causal --padding-ratio 0.3
run large_batch_causal         --batch-size 256 --seq-len 128  --causal
run large_batch_causal_padded  --batch-size 256 --seq-len 128  --causal --padding-ratio 0.3
