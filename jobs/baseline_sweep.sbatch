#!/bin/bash
#SBATCH --job-name=baseline_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/runs/%j.out

set -euo pipefail

clocks() {
    nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw \
        --format=csv,noheader
}

run() {
    local name="$1"; shift
    echo "=== SHAPE: $name ==="
    echo "[clocks before] $(clocks)"
    apptainer exec --nv --cleanenv \
        --bind /scratch/work:/work \
        /scratch/kernel.sif \
        python3 /work/benchmark.py "$@"
    echo "[clocks after]  $(clocks)"
    echo
}

# Regime coverage per CLAUDE.md dispatch table.
run tiny             --batch-size 1   --seq-len 64
run default           --batch-size 8   --seq-len 128
run long_seq          --batch-size 8   --seq-len 1024
run large_batch       --batch-size 256 --seq-len 128
run default_padded    --batch-size 8   --seq-len 128 --padding-ratio 0.3
run default_causal    --batch-size 8   --seq-len 128 --causal
