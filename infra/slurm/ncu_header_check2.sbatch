#!/bin/bash
#SBATCH --job-name=ncu_header_check2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -euo pipefail

METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed"

apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    ncu --metrics "$METRICS" --csv --target-processes all --launch-count 3 \
    python3 /work/torch_transformer_benchmark.py --batch-size 8 --seq-len 128 \
        --accuracy-trials 1 --warmup 1 --repeats 1 --benchmark-rounds 1 \
    2>/dev/null \
    | grep -n -A 3 '"ID"' | head -20
