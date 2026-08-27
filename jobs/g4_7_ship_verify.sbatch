#!/bin/bash
#SBATCH --job-name=g4_7_ship
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

# G4.7 ship verification -- fused ffn_in + exact-erf GELU in the CAUSAL path.
#
#   PASS 1 (BEFORE)  = benchmark.py at HEAD (aa12749; byte-identical to the
#                      shipped 0691311 -- the WIP commit did not touch it).
#                      5 accuracy trials: it is already the verified shipped
#                      model, this pass only fixes a same-job, same-clocks
#                      speedup baseline.
#   PASS 2 (AFTER58) = working tree, G4_7_FFN_CFG=58 -- ACCF32 fused GELU,
#                      PRECISION-NEUTRAL (fp32 accumulate + bit-identical
#                      activation). 40 trials. THIS is the shippable variant;
#                      its causal accuracy must match BEFORE to the ~4th digit
#                      and it must be faster on at least one judged shape.
#   PASS 3 (AFTER51) = working tree, G4_7_FFN_CFG=51 -- FP16-accumulate fused
#                      GELU. Runs ONLY on the K=128 causal rows (d_model=128);
#                      completes the accuracy record for whether FP16
#                      accumulation on the *ffn_in* GEMM (K=128, vs the K>=512
#                      attention GEMMs step 41 closed) fits the causal budget.
#                      40 trials. Not a ship candidate unless it passes here.
#
# Shape groups:
#   G-WIN   causal shapes the microbench (run132) predicts a win at: the
#           project's internal d512/ffn2048 sweep, long_seq + large_batch.
#   G-JUDGE official all-causal matrix rows where the FFN gate engages
#           (tok>=8192, ffn_dim%128==0): rows 1, 6, 8, 13.
#   G-CTRL  gate-off / untouched-path controls that MUST come back unchanged:
#           official row 4 (tok<8192), row 7 (ffn_dim=32), and two non-causal
#           shapes (G4.7 is not wired into _optimized_forward at all).

set -uo pipefail

mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build

cd /scratch/work
# Pinned to 0691311 (G4.3 ship) -- the last commit before any G4.7 work.
# benchmark.py there is byte-identical to HEAD/aa12749 (the WIP commit did
# not touch it); pinning a SHA lets G4.7 be committed without breaking this.
git show 0691311:benchmark.py > /scratch/work/_before_benchmark.py

run() {
    local tag="$1"; shift
    echo "=== ${tag} ==="
    apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
        python3 "$@" --benchmark-on-failure || true
    echo
}

# ---------------------------------------------------------------------------
for pass in BEFORE AFTER58 AFTER51; do
  case "$pass" in
    BEFORE)  S=/work/_before_benchmark.py; T=5;  export G4_7_FFN_CFG=58 ;;
    AFTER58) S=/work/benchmark.py;         T=40; export G4_7_FFN_CFG=58 ;;
    AFTER51) S=/work/benchmark.py;         T=40; export G4_7_FFN_CFG=51 ;;
  esac
  echo "##################### PASS: $pass ($S, T=$T, G4_7_FFN_CFG=$G4_7_FFN_CFG) #####################"

  if [ "$pass" = "AFTER51" ]; then
    # FP16-accumulate: only meaningful on K=128 (d_model=128) causal rows.
    run "$pass off_row1_tok8192"   "$S" --accuracy-trials "$T" --causal --batch-size 64    --d-model 128 --heads 4 --seq-len 128  --layers 4 --ffn-dim 128
    run "$pass off_row13_tok65536" "$S" --accuracy-trials "$T" --causal --batch-size 64    --d-model 128 --heads 4 --seq-len 1024 --layers 4 --ffn-dim 128
    run "$pass off_row6_tok1.28M"  "$S" --accuracy-trials "$T" --causal --batch-size 10000 --d-model 128 --heads 4 --seq-len 128  --layers 4 --ffn-dim 128
    continue
  fi

  # -- G-WIN: internal d512/ffn2048 causal sweep (microbench predicts x1.4-1.5)
  run "$pass int_long_seq_causal"    "$S" --accuracy-trials "$T" --causal --batch-size 8   --seq-len 1024
  run "$pass int_large_batch_causal" "$S" --accuracy-trials "$T" --causal --batch-size 256 --seq-len 128

  # -- G-JUDGE: official all-causal matrix rows where the gate engages
  run "$pass off_row1_tok8192"   "$S" --accuracy-trials "$T" --causal --batch-size 64    --d-model 128  --heads 4 --seq-len 128  --layers 4 --ffn-dim 128
  run "$pass off_row8_tok8192"   "$S" --accuracy-trials "$T" --causal --batch-size 64    --d-model 1024 --heads 4 --seq-len 128  --layers 4 --ffn-dim 1024
  run "$pass off_row13_tok65536" "$S" --accuracy-trials "$T" --causal --batch-size 64    --d-model 128  --heads 4 --seq-len 1024 --layers 4 --ffn-dim 128
  run "$pass off_row6_tok1.28M"  "$S" --accuracy-trials "$T" --causal --batch-size 10000 --d-model 128  --heads 4 --seq-len 128  --layers 4 --ffn-dim 128

  # -- G-CTRL: must be unchanged
  run "$pass off_row4_gateoff"   "$S" --accuracy-trials "$T" --causal --batch-size 16 --d-model 128 --heads 4 --seq-len 128 --layers 4 --ffn-dim 128
  run "$pass off_row7_ffn32"     "$S" --accuracy-trials "$T" --causal --batch-size 64 --d-model 32  --heads 4 --seq-len 128 --layers 4 --ffn-dim 32
  run "$pass nc_long_seq"        "$S" --accuracy-trials "$T" --batch-size 8   --seq-len 1024
  run "$pass nc_large_batch"     "$S" --accuracy-trials "$T" --batch-size 256 --seq-len 128
done
