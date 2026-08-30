#!/bin/bash
# G8.1 -- is FlashAttention-3 a possible optimization here? FA3 needs Hopper
# (wgmma + TMA + async warp specialisation); this is Ada sm_89. Verifies the
# hardware gate, records what PyTorch actually dispatches, and measures
# attention's share of the forward on all 14 shapes so the ceiling on ANY
# faster attention kernel is a number rather than a guess.
#
# ONE SHAPE PER PROCESS, deliberately. Job 230 drove all 14 through a single
# interpreter: dynamo blew its recompile limit so later shapes silently fell
# back to eager, and CUDA-graph replay collapsed the kernel->op correlation the
# split depends on, reporting a meaningless "SDPA 0.0%, other 100%".
#SBATCH --job-name=g8_1_fa3_evaluation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail
mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN() { apptainer exec --nv --cleanenv --bind /scratch/work:/work \
        --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        /scratch/kernel.sif python3 /work/experiments/g8_1_fa3_evaluation.py "$@"; }

RUN --caps
echo
echo "== 2. attention_share (one shape per process) =="
for r in row01 row02 row03 row04 row05 row06 row07 row08 row09 row10 \
         row11 row12 row13 row14; do
    OUT=$(RUN --shape "$r" 2>&1)
    echo "$OUT" | grep -E "^SHARE |^  row" || {
        echo "  $r FAILED:"; echo "$OUT" | tail -4 | sed 's/^/      /'; }
done

J="${SLURM_JOB_ID}"
RAW="/scratch/techjam2/runs/${J}.out"
apptainer exec --nv --cleanenv --bind /scratch/work:/work --bind /scratch/techjam2:/host \
    /scratch/kernel.sif python3 /work/experiments/g8_1_fa3_evaluation.py \
    --amdahl "/host/runs/${J}.out"

cp "$RAW" "/scratch/work/results/logs/g8_1_fa3_evaluation_run${J}.log" 2>/dev/null || true
