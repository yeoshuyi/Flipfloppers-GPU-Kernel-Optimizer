#!/bin/bash
#SBATCH --job-name=ncu_profile_ss
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/runs/%j.out

# Same as jobs/profile.sbatch, but with real warmup (5 calls) BEFORE the
# profiled window and a much larger --launch-count, so ncu's window lands
# on steady-state CUDA-graph REPLAY kernels rather than the one-time
# torch.compile/cudagraph-capture flood (which emits many more kernels
# than a single steady replay -- autotuning candidates, capture-time
# bookkeeping). jobs/profile.sbatch's original --launch-count 120 with
# --warmup 1 likely captured mostly first-call compile/capture overhead,
# not the replay behavior we actually want to characterize post-G2.4.
#
# Usage: sbatch jobs/profile_steady_state.sbatch [--dtype tf32|bf16|fp8] -- <shape args>

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
    ncu --metrics "$METRICS" --csv --target-processes all \
    --launch-skip 4000 --launch-count 200 \
    python3 /work/benchmark.py "$@" \
        --accuracy-trials 1 --warmup 20 --repeats 5 --benchmark-rounds 1 \
    2>/dev/null \
    | apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
      python3 /work/tools/parse_ncu.py --dtype "$DTYPE"
