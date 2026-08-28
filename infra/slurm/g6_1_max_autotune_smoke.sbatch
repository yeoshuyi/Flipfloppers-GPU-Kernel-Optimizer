#!/bin/bash
#SBATCH --job-name=g6_1_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -euo pipefail

run() {
    local name="$1"; shift
    echo "=== SHAPE: $name ==="
    /usr/bin/time -f "wall=%es" apptainer exec --nv --cleanenv \
        --bind /scratch/work:/work \
        /scratch/kernel.sif \
        python3 /work/benchmark.py "$@"
    echo
}

run tiny            --batch-size 1 --seq-len 64
run default         --batch-size 8 --seq-len 128
run default_causal  --batch-size 8 --seq-len 128 --causal
