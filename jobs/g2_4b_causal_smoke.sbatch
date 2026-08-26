#!/bin/bash
#SBATCH --job-name=g2_4b_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/runs/%j.out

set -euo pipefail

run() {
    local name="$1"; shift
    echo "=== SHAPE: $name ==="
    apptainer exec --nv --cleanenv \
        --bind /scratch/work:/work \
        /scratch/kernel.sif \
        python3 /work/benchmark.py "$@"
    echo
}

run default_causal --batch-size 8 --seq-len 128 --causal
run causal_padded   --batch-size 8 --seq-len 128 --causal --padding-ratio 0.3
