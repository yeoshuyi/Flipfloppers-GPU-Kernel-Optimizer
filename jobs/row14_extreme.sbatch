#!/bin/bash
# Official row 14 -- the EXTREME shape, run on explicit user request.
#   B=32, QKV/d_model=1024, heads=16, seq_len=100000, layers=2, causal, ffn=1024
#
# This row is excluded from jobs/official_causal_sweep.sbatch because the
# BaselineTransformer's manual attention materialises a full [S,S] score
# tensor per (batch, head): 32*16*100000*100000*4 B ~= 2 exabytes. Q/K/V are
# ~13 GiB each and the fp32 input x is ~12.2 GiB, so an OOM before any
# attention math is the expected outcome on a 24 GiB RTX 4090. Running it to
# get the real message.
#
# benchmark.py runs the accuracy check (needs a baseline forward) BEFORE the
# perf benchmark, so if the baseline OOMs the run stops there. A second
# invocation with --benchmark-on-failure is added so we still see whether the
# optimized (flash-SDPA, O(S) memory) path can stand up on its own.

#SBATCH --job-name=row14_extreme
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

ARGS="--causal --batch-size 32 --d-model 1024 --heads 16 --seq-len 100000 --layers 2 --ffn-dim 1024"

echo "############## RUN 1: standard (baseline accuracy-check + both timed) ##############"
apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
    python3 /work/benchmark.py $ARGS 2>&1 || true

echo
echo "############## RUN 2: --benchmark-on-failure (time whatever can run) ##############"
apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
    python3 /work/benchmark.py $ARGS --benchmark-on-failure 2>&1 || true

J="${SLURM_JOB_ID}"
cp "/scratch/techjam2/runs/${J}.out" "/scratch/work/results/logs/row14_extreme_run${J}.log" 2>/dev/null || true
