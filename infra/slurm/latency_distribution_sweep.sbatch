#!/bin/bash
# Latency-distribution sweep -- NOT a scoring run.
#
# Runs tools/latency_distribution_sweep.py over the official 14-row causal
# matrix (CLAUDE.md -> "OFFICIAL CAUSAL EVALUATION MATRIX"), collecting raw
# per-iteration latency samples per (shape, variant) so the *distribution* of
# latency at each shape can be studied (see tools/plot_latency_distribution.py
# for turning the resulting JSON into a chart).
#
# --exclusive --gres=gpu:1 to keep this off the shared node's GPU while other
# jobs run (see docs/SETUP.md); --time is generous vs. the 13-row scoring
# sweep's 30 minutes since this runs several times more timed iterations per
# shape (default 500/shape vs. ~150/shape/variant in run_eval.sh) plus row
# 14's slow chunked optimized path.
#
# Runs with the host python3 venv directly rather than through the apptainer
# image: /scratch/kernel.sif does not currently exist on this node (never
# built, or since removed), and this is an exploratory diagnostic, not a
# scoring run judges will re-execute, so container-level reproducibility
# doesn't apply. The host venv's torch/triton versions match requirements.txt
# exactly (torch==2.13.0+cu130, triton==3.7.1) and its nvcc (13.1.115) matches
# the container's, so csrc/ custom-kernel JIT compilation works identically.

# Rows 1-13 and row 14 run as two SEPARATE python processes, sequentially, not
# one process handling both: row 14's ~12 GiB kv-cache allocation is tight
# enough that a still-resident prior process's CUDA context (a few GiB of
# cuBLAS/cuDNN handles and compile caches survive even torch.cuda.empty_cache())
# is enough to OOM it (confirmed empirically). Only a process that has actually
# exited fully frees its share of the card back to the driver.

#SBATCH --job-name=latency_dist_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

cd /scratch/work
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT13="results/artifacts/latency_distribution_${STAMP}_rows1-13.json"
OUT14="results/artifacts/latency_distribution_${STAMP}_row14.json"

python3 tools/latency_distribution_sweep.py --out "$OUT13" "$@"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 tools/latency_distribution_sweep.py --only-row 14 --out "$OUT14" "$@"

echo "latency_distribution_sweep: wrote $OUT13 and $OUT14"
echo "latency_distribution_sweep: plot with:"
echo "  python3 tools/plot_latency_distribution.py $OUT13 $OUT14"
