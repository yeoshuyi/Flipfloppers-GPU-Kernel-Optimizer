#!/bin/bash
#SBATCH --job-name=ncu_profile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/runs/%j.out

# Usage: sbatch jobs/profile.sbatch [--dtype tf32|bf16|fp8] -- <benchmark.py shape args>
# Example: sbatch jobs/profile.sbatch --dtype tf32 -- --batch-size 8 --seq-len 128
#
# Runs ncu with a curated metric list (jobs/ncu_discover.sbatch,
# jobs/ncu_discover2.sbatch) against a MINIMAL invocation of benchmark.py
# (1 accuracy trial, 1 warmup, 1 repeat, 1 round -- kernel-level structural
# facts don't need the full timing sweep, and ncu's per-kernel profiling
# overhead makes profiling the full sweep impractically slow), pipes
# directly through tools/parse_ncu.py. The raw CSV is NEVER written to a
# file or otherwise allowed to reach this job's stdout -- only the parser's
# compact JSON does (docs/SETUP.md: "Parse ncu to JSON in tools/ -- never
# let raw output reach an agent's context").

set -euo pipefail

DTYPE="tf32"
if [[ "${1:-}" == "--dtype" ]]; then
    DTYPE="$2"; shift 2
fi
if [[ "${1:-}" == "--" ]]; then
    shift
fi

METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active,smsp__warp_issue_stalled_barrier_per_warp_active,smsp__warp_issue_stalled_mio_throttle_per_warp_active,smsp__warp_issue_stalled_short_scoreboard_per_warp_active,smsp__warp_issue_stalled_not_selected_per_warp_active,smsp__warp_issue_stalled_wait_per_warp_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,sm__inst_executed_pipe_tensor_op_hmma_v2.sum"

apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    ncu --metrics "$METRICS" --csv --target-processes all --launch-count 120 \
    python3 /work/benchmark.py "$@" \
        --accuracy-trials 1 --warmup 1 --repeats 1 --benchmark-rounds 1 \
    2>/dev/null \
    | apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
      python3 /work/tools/parse_ncu.py --dtype "$DTYPE"
