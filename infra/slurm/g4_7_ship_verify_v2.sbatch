#!/bin/bash
#SBATCH --job-name=g4_7_shv2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

# G4.7 ship verification v2 -- after tightening _ensure_ffn_plan to
# d_model>=512 AND ffn_dim>=2048 (the shapes run139 proved a precision-neutral
# win at). Focused: only the shapes the gate now engages at, plus controls
# that must be byte-identical.
#
#   BEFORE  = 0691311 benchmark.py (pre-G4.7), 5 trials, speedup baseline only.
#   AFTER   = working tree, G4_7_FFN_CFG=58, 40 trials.
#
# Engages:  default d512/ffn2048 causal at long_seq (tok 8192) and
#           large_batch (tok 32768).
# Controls: default_causal (d512, tok 1024 -- below token gate),
#           official row 1 (d128 -- below shape gate),
#           nc_large_batch (non-causal -- G4.7 not in that path at all).

set -uo pipefail
mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build
cd /scratch/work
git show 0691311:benchmark.py > /scratch/work/_before_benchmark.py

run() {
    local tag="$1"; shift
    echo "=== ${tag} ==="
    apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
        python3 "$@" --benchmark-on-failure || true
    echo
}

for pass in BEFORE AFTER; do
  if [ "$pass" = "BEFORE" ]; then S=/work/_before_benchmark.py; T=5; else S=/work/benchmark.py; T=40; fi
  export G4_7_FFN_CFG=58
  echo "##################### PASS: $pass ($S, T=$T) #####################"
  run "$pass long_seq_causal"    "$S" --accuracy-trials "$T" --causal --batch-size 8   --seq-len 1024
  run "$pass large_batch_causal" "$S" --accuracy-trials "$T" --causal --batch-size 256 --seq-len 128
  run "$pass default_causal"     "$S" --accuracy-trials "$T" --causal --batch-size 8   --seq-len 128
  run "$pass off_row1_d128"      "$S" --accuracy-trials "$T" --causal --batch-size 64  --d-model 128 --heads 4 --seq-len 128 --layers 4 --ffn-dim 128
  run "$pass nc_large_batch"     "$S" --accuracy-trials "$T" --batch-size 256 --seq-len 128
done
