#!/bin/bash
#SBATCH --job-name=g1_6_decomp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

mkdir -p /scratch/work/.ext_build

# G1.6 arena ONLY -- G2.3's L2 pin disabled, to tell "the arena repointing
# costs something" apart from "the persisting-L2 reservation costs something".
# Causal shapes are omitted: they bypass _ensure_folded_weights entirely and
# already served as the unchanged control in the combined sweep.
run() {
    local name="$1"; shift
    echo "=== SHAPE: $name (arena only, KOPT_L2_PERSIST=0) ==="
    apptainer exec --nv --cleanenv \
        --bind /scratch/work:/work \
        /scratch/kernel.sif \
        bash -c '
          export TMPDIR=/work/.ext_build
          export TORCH_EXT_BUILD_DIR=/work/.ext_build
          export KOPT_L2_PERSIST=0
          exec python3 /work/benchmark.py "$@"
        ' _ "$@"
    echo
}

run tiny             --batch-size 1   --seq-len 64
run default            --batch-size 8   --seq-len 128
run long_seq           --batch-size 8   --seq-len 1024
run large_batch        --batch-size 256 --seq-len 128
run default_padded     --batch-size 8   --seq-len 128 --padding-ratio 0.3
run long_seq_padded    --batch-size 8   --seq-len 1024 --padding-ratio 0.3

echo "=== probe: L2 window effect per regime ==="
apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    bash -c '
      export TMPDIR=/work/.ext_build
      export TORCH_EXT_BUILD_DIR=/work/.ext_build
      python3 /work/probes/g2_3_l2_persist_probe.py
    '
