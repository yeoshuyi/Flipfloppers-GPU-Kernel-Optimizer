#!/bin/bash
#SBATCH --job-name=g4_7_entry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

# Confirm the generated torch_transformer_benchmark.py behaves identically to
# benchmark.py -- same accuracy summary, speedup within noise -- on one
# official-matrix causal row and one d512/ffn2048 causal shape (where G4.7
# engages).

set -uo pipefail
mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build
export G4_7_FFN_CFG=58

run() {
    local label="$1"; local script="$2"; shift 2
    echo "=== ${label} :: ${script} ==="
    apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
        python3 "/work/${script}" --accuracy-trials 10 --benchmark-on-failure "$@" || true
    echo
}

for shape in \
    "official_row1 --causal --batch-size 64 --d-model 128 --heads 4 --seq-len 128 --layers 4 --ffn-dim 128" \
    "d512_long_seq_causal --causal --batch-size 8 --seq-len 1024" ; do
  name="${shape%% *}"; args="${shape#* }"
  run "$name" benchmark.py $args
  run "$name" torch_transformer_benchmark.py $args
done
