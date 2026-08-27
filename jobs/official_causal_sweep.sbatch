#!/bin/bash
# Official Causal Evaluation Matrix -- SCAFFOLD, NOT YET COMPLETE.
#
# The 14-row official causal test-vector table (Batch Size | QKV Dim |
# Heads | Seq Len | Layers | Causal | FFN Dim) relayed this session did not
# survive context compaction intact -- only 2 of 14 rows are known here.
# This script is a placeholder: it runs the 2 confirmed-in-context rows
# plus the project's own standard causal shapes (as an interim substitute
# for the missing 12), and is meant to be replaced wholesale once the full
# table is reconfirmed. See CLAUDE.md's "OFFICIAL CAUSAL EVALUATION MATRIX"
# section for the provenance note and the B=10000/S=100000 flag below.
#
# DO NOT run the "extreme" row as literally specified without confirming
# it first: B=10000, S=100000 causal attention needs a S x S=10^10-element
# score tensor per (batch, head) -- 40GB at FP32 for ONE pair alone, on a
# 24GB RTX 4090. It is commented out below rather than silently run (which
# would just CUDA-OOM and burn a shared-node exclusive job slot for
# nothing) -- uncomment only after the shape is reconfirmed, and even then
# expect it may need to be run with a corrected, smaller S and/or B.

#SBATCH --job-name=off_causal_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

run() {
    local name="$1"; shift
    echo "=== SHAPE: $name ==="
    apptainer exec --nv --cleanenv \
        --bind /scratch/work:/work \
        /scratch/kernel.sif \
        python3 /work/benchmark.py --causal "$@" || true
    echo
}

# --- Confirmed rows from the relayed table (2 of 14) -----------------------
# Row: default-ish causal shape (matches the project's own default_causal).
run official_row_1     --batch-size 8    --seq-len 128

# Row: the flagged extreme -- see header note. NOT run as-is.
# run official_row_extreme --batch-size 10000 --seq-len 100000 --layers 2

# --- Interim substitutes for the 12 missing rows ----------------------------
# Standard project causal shapes, run here only until the real table lands.
run tiny_causal         --batch-size 1   --seq-len 64
run long_seq_causal     --batch-size 8   --seq-len 1024
run large_batch_causal  --batch-size 256 --seq-len 128
run padded_causal       --batch-size 8   --seq-len 128  --padding-ratio 0.3
