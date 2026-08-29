#!/bin/bash
#SBATCH --job-name=g5_2_ship
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

# T3 ship verification -- _ensure_ffn_plan extended with a second admit regime
# (tok >= 2^19, any ffn_dim >= 128) so G4.7's fused ffn_in+GELU engages on the
# deeply memory-bound official row 6 (B=10000, d128, tok 1.28M), where g5_2
# (run148) measured cfg 58 at x1.66 vs the shipped chain -- precision-neutral.
#
#   BEFORE = benchmark.py at 09dee91 (G4.7 gate = d512/ffn2048 only).
#   AFTER  = working tree (+ the tok>=2^19 admit regime). 40 trials.
#
# Engages: official row 6.
# Controls (gate OFF / other path, must be unchanged):
#   official row 13 (tok 65536 < 2^19), official row 1, official row 8,
#   int_large_batch_causal (d512 path -- G4.7 regime 1, regression check),
#   nc_large_batch (non-causal).

set -uo pipefail
mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build
cd /scratch/work
git show 09dee91:benchmark.py > /scratch/work/_before_benchmark.py

run() {
    local tag="$1"; shift
    echo "=== ${tag} ==="
    apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
        python3 "$@" --benchmark-on-failure || true
    echo
}

for pass in BEFORE AFTER; do
  if [ "$pass" = "BEFORE" ]; then S=/work/_before_benchmark.py; T=5; else S=/work/torch_transformer_benchmark.py; T=40; fi
  echo "##################### PASS: $pass ($S, T=$T) #####################"
  run "$pass off_row6_tok1.28M"   "$S" --accuracy-trials "$T" --causal --batch-size 10000 --d-model 128  --heads 4 --seq-len 128  --layers 4 --ffn-dim 128
  run "$pass off_row13_tok65536"  "$S" --accuracy-trials "$T" --causal --batch-size 64    --d-model 128  --heads 4 --seq-len 1024 --layers 4 --ffn-dim 128
  run "$pass off_row1_tok8192"    "$S" --accuracy-trials "$T" --causal --batch-size 64    --d-model 128  --heads 4 --seq-len 128  --layers 4 --ffn-dim 128
  run "$pass off_row8_d1024"      "$S" --accuracy-trials "$T" --causal --batch-size 64    --d-model 1024 --heads 4 --seq-len 128  --layers 4 --ffn-dim 1024
  run "$pass int_large_batch_causal" "$S" --accuracy-trials "$T" --causal --batch-size 256 --seq-len 128
  run "$pass nc_large_batch"      "$S" --accuracy-trials "$T" --batch-size 256 --seq-len 128
done
