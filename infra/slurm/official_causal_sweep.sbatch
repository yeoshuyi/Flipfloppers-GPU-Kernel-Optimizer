#!/bin/bash
# Official Causal Evaluation Matrix -- full 14-row table, from CLAUDE.md's
# "OFFICIAL CAUSAL EVALUATION MATRIX" section (synced from the project
# owner's authoritative copy). Replaces the earlier placeholder version of
# this script, which only had 2 of 14 rows confirmed (lost to context
# compaction) and substituted the project's own standard causal shapes for
# the rest.
#
# Column mapping to benchmark.py flags:
#   Batch Size -> --batch-size   QKV Dim -> --d-model   Heads -> --heads
#   Seq Len    -> --seq-len      Layers  -> --layers     FFN Dim -> --ffn-dim
#   Causal is TRUE for all 14 rows -> --causal always passed.
#
# Regime label per row computed from CLAUDE.md's own REGIME DISPATCH table
# (tok = batch*seq_len; tiny: tok<128; long-seq: seq_len>=1024 checked
# before large-batch; large-batch: tok>16384; else default).
#
# Row 14 (B=32, QKV=1024, Heads=16, S=100000, Layers=2, FFN=1024) is NOT
# run automatically -- see the guard block below for the memory math. It is
# left in the script, correctly parameterized, for deliberate manual use.

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

# --- Row 1: default -----------------------------------------------------
run official_row_1_default    --batch-size 64    --d-model 128  --heads 4  --seq-len 128    --layers 4 --ffn-dim 128

# --- Row 2: default (B=1, smallest batch but tok=128 == boundary, not <128)
run official_row_2_default    --batch-size 1     --d-model 128  --heads 4  --seq-len 128    --layers 4 --ffn-dim 128

# --- Row 3: default -------------------------------------------------------
run official_row_3_default    --batch-size 4     --d-model 128  --heads 4  --seq-len 128    --layers 4 --ffn-dim 128

# --- Row 4: default -------------------------------------------------------
run official_row_4_default    --batch-size 16    --d-model 128  --heads 4  --seq-len 128    --layers 4 --ffn-dim 128

# --- Row 5: default (tok=16384 == boundary, not >16384) --------------------
run official_row_5_default    --batch-size 128   --d-model 128  --heads 4  --seq-len 128    --layers 4 --ffn-dim 128

# --- Row 6: large-batch (tok=1,280,000) -------------------------------------
run official_row_6_large_batch --batch-size 10000 --d-model 128  --heads 4  --seq-len 128    --layers 4 --ffn-dim 128

# --- Row 7: default, narrow QKV/FFN dim -------------------------------------
run official_row_7_default    --batch-size 64    --d-model 32   --heads 4  --seq-len 128    --layers 4 --ffn-dim 32

# --- Row 8: default, wide QKV/FFN dim ---------------------------------------
run official_row_8_default    --batch-size 64    --d-model 1024 --heads 4  --seq-len 128    --layers 4 --ffn-dim 1024

# --- Row 9: default, single head --------------------------------------------
run official_row_9_default    --batch-size 64    --d-model 128  --heads 1  --seq-len 128    --layers 4 --ffn-dim 128

# --- Row 10: default, 2 heads ------------------------------------------------
run official_row_10_default   --batch-size 64    --d-model 128  --heads 2  --seq-len 128    --layers 4 --ffn-dim 128

# --- Row 11: default, 16 heads -----------------------------------------------
run official_row_11_default   --batch-size 64    --d-model 128  --heads 16 --seq-len 128    --layers 4 --ffn-dim 128

# --- Row 12: default, short seq (S=32) ----------------------------------------
run official_row_12_default   --batch-size 64    --d-model 128  --heads 4  --seq-len 32     --layers 4 --ffn-dim 128

# --- Row 13: long-seq (S=1024) ------------------------------------------------
run official_row_13_long_seq  --batch-size 64    --d-model 128  --heads 4  --seq-len 1024   --layers 4 --ffn-dim 128

# --- Row 14: long-seq, EXTREME -- NOT run automatically -----------------------
# B=32, QKV=1024, Heads=16, S=100000, Layers=2, FFN=1024, causal=TRUE.
#
# Memory math against a 24GiB RTX 4090, FP32:
#   * Baseline's manual attention path materializes a full S x S score
#     tensor per (batch, head): 32 * 16 * 100000 * 100000 * 4 bytes
#     ~= 2.05 EXABYTES for one tensor alone -- not a slow run, an immediate,
#     guaranteed allocation failure the instant BaselineTransformer's
#     manual matmul/mask/softmax/matmul loop runs.
#   * Even before that: Q/K/V alone (B*heads*S*head_dim*4 bytes each) are
#     ~13.1 GiB EACH (~39 GiB combined), and the input activation tensor
#     x (B*S*d_model*4 bytes) is ~12.2 GiB by itself -- both already at or
#     beyond the card's 24 GiB before any attention math happens.
#
# Running this row as-is will CUDA-OOM immediately and burn a shared-node
# --exclusive GPU slot for nothing (see the slurm shared-node note in
# docs/SETUP.md). It is intentionally left commented out. If this row is
# ever run for real, it needs a deliberate, separate decision first (e.g.
# a memory-feasibility check, and likely only makes sense against the
# optimized/flash path in isolation, not BaselineTransformer's manual
# attention, which cannot run this shape at all regardless of any kernel
# optimization).
# run official_row_14_extreme --batch-size 32 --d-model 1024 --heads 16 --seq-len 100000 --layers 2 --ffn-dim 1024
