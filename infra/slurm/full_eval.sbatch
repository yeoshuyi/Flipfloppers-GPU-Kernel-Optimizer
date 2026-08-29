#!/bin/bash
# Full end-to-end evaluation over the official 14-row causal matrix, run the
# way a grader would: ./run_eval.sh drives
# `python3 /work/torch_transformer_benchmark.py --causal <shape args>` per row
# inside the pinned Apptainer image, with NO extra flags (harness defaults:
# atol 0.002, rtol 0.02, 5 accuracy trials, 20 warmup, 100 repeats, 3 rounds).
#
# Rows 1-13 are scored by the harness. Row 14 is additionally ATTEMPTED through
# the harness here, so the OOM is documented rather than asserted, and then run
# through the capability probe (RUN_ROW14=1).
#SBATCH --job-name=full_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail
mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build

cd /scratch/work

echo "################ ROWS 1-13 + row-14 probe : ./run_eval.sh ################"
RUN_ROW14=1 ./run_eval.sh || true

echo
echo "################ ROW 14 THROUGH THE HARNESS ITSELF ################"
echo "# expected: the FP32 reference OOMs before either model runs."
ARGS="--causal --batch-size 32 --d-model 1024 --heads 16 --seq-len 100000 --layers 2 --ffn-dim 1024"
apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
    python3 /work/torch_transformer_benchmark.py $ARGS 2>&1 | tail -25 || true

J="${SLURM_JOB_ID}"
cp "/scratch/techjam2/runs/${J}.out" \
   "/scratch/work/results/logs/full_eval_run${J}.log" 2>/dev/null || true
