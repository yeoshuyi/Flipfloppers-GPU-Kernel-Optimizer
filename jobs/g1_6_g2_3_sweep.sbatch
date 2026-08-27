#!/bin/bash
#SBATCH --job-name=g1_6_g2_3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

mkdir -p /scratch/work/.ext_build

# Warm the extension build once, outside the timed runs, so no shape pays the
# ~30 s nvcc/g++ cost. cpp_extension.load() caches into the build directory.
apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    bash -c '
      export TMPDIR=/work/.ext_build
      export TORCH_EXT_BUILD_DIR=/work/.ext_build
      mkdir -p "$TMPDIR"
      python3 -c "
import os, torch
from torch.utils.cpp_extension import load
ext = load(name=\"l2_persist\", sources=[\"/work/csrc/l2_persist.cpp\"],
           build_directory=\"/work/.ext_build\", with_cuda=True, verbose=False)
print(\"ext ok:\", ext.l2_cache_size(), ext.persisting_l2_max_size(),
      ext.access_policy_max_window_size())
"
    '

run() {
    local name="$1"; shift
    echo "=== SHAPE: $name ==="
    apptainer exec --nv --cleanenv \
        --bind /scratch/work:/work \
        /scratch/kernel.sif \
        bash -c '
          export TMPDIR=/work/.ext_build
          export TORCH_EXT_BUILD_DIR=/work/.ext_build
          exec python3 /work/benchmark.py "$@"
        ' _ "$@"
    echo
}

run tiny             --batch-size 1   --seq-len 64
run default            --batch-size 8   --seq-len 128
run long_seq           --batch-size 8   --seq-len 1024
run large_batch        --batch-size 256 --seq-len 128
run default_padded     --batch-size 8   --seq-len 128 --padding-ratio 0.3
run default_causal     --batch-size 8   --seq-len 128 --causal
run causal_padded      --batch-size 8   --seq-len 128 --causal --padding-ratio 0.3
run long_seq_padded    --batch-size 8   --seq-len 1024 --padding-ratio 0.3
