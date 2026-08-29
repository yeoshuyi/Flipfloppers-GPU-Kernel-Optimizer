#!/bin/bash
#SBATCH --job-name=g5_4_ship
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

# T3(b) ship verification -- the OUT-PARAMETER fused ffn_in+GELU op for the
# memory-bound regime (tok >= 2^19), verified to engage through CUDA-graph
# capture (g5_3, run154: ws_gemm_kernel present in the d128 captured trace).
#
# Clean matched comparison: BOTH passes run the SAME working-tree benchmark.py,
# 40 accuracy trials each -- only G4_7_FFN_CFG differs, so the speedup isolates
# exactly "fused row-6 path off vs on" (no 5-vs-40-trial confound).
#   BEFORE = G4_7_FFN_CFG=-1  (fused path disabled everywhere)
#   AFTER  = G4_7_FFN_CFG=58  (default)
#
# Engages: official row 6 (d128, tok 1.28M) via the out-param op;
#          int_large_batch_causal (d512 -- regime 1, the RETURNING op) as a
#          regression check.
# Controls (gate off): official rows 13 (tok 65536 < 2^19), 1, 8; nc_large_batch.

set -uo pipefail
mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build
cd /scratch/work
S=/work/torch_transformer_benchmark.py

run() {
    local tag="$1"; shift
    echo "=== ${tag} ==="
    apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
        python3 "$S" --accuracy-trials 40 --benchmark-on-failure "$@" || true
    echo
}

for pass in BEFORE AFTER; do
  if [ "$pass" = "BEFORE" ]; then export G4_7_FFN_CFG=-1; else export G4_7_FFN_CFG=58; fi
  echo "##################### PASS: $pass (G4_7_FFN_CFG=$G4_7_FFN_CFG) #####################"
  run "$pass off_row6_tok1.28M"       --causal --batch-size 10000 --d-model 128  --heads 4 --seq-len 128  --layers 4 --ffn-dim 128
  run "$pass off_row13_tok65536"      --causal --batch-size 64    --d-model 128  --heads 4 --seq-len 1024 --layers 4 --ffn-dim 128
  run "$pass off_row1_tok8192"        --causal --batch-size 64    --d-model 128  --heads 4 --seq-len 128  --layers 4 --ffn-dim 128
  run "$pass off_row8_d1024"          --causal --batch-size 64    --d-model 1024 --heads 4 --seq-len 128  --layers 4 --ffn-dim 1024
  run "$pass int_large_batch_causal"  --causal --batch-size 256   --seq-len 128
  run "$pass nc_large_batch"          --batch-size 256 --seq-len 128
done
