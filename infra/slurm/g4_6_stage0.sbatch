#!/bin/bash
#SBATCH --job-name=g4_6_st0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

mkdir -p /scratch/work/.ext_build

apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    bash -c '
      export TMPDIR=/work/.ext_build
      export TORCH_EXT_BUILD_DIR=/work/.ext_build
      mkdir -p "$TMPDIR"

      echo "===== raw nvcc pre-check (CUTLASS headers WITHOUT torch headers) ====="
      echo "--- nvcc version ---"; nvcc --version
      T0=$SECONDS
      nvcc -c /work/csrc/g4_6_cutlass_stock.cu -o "$TMPDIR/g4_6_raw.o" \
          -std=c++17 \
          -I/work/.cutlass/include -I/work/.cutlass/tools/util/include \
          -gencode=arch=compute_89,code=sm_89 \
          --expt-relaxed-constexpr --expt-extended-lambda \
          -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -Xptxas -v 2>&1 | tail -60
      RC=${PIPESTATUS[0]}
      echo "raw nvcc exit: $RC   wall: $((SECONDS - T0)) s"

      echo
      echo "===== torch cpp_extension build + gate 0b ====="
      python3 /work/experiments/g4_6_cutlass_stage0.py
      echo "probe exit: $?"
    '
