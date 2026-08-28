#!/bin/bash
#SBATCH --job-name=g6_6_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -uo pipefail

mkdir -p /scratch/work/.ext_build

# Warm the extension build once, outside the timed runs, so no shape pays the
# nvcc/g++ cost inside its own measurement.
apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
    bash -c '
      export TMPDIR=/work/.ext_build
      export TORCH_EXT_BUILD_DIR=/work/.ext_build
      mkdir -p "$TMPDIR"
      python3 -c "import sys; sys.path.insert(0,\"/work\"); import benchmark; print(\"ext warm:\", benchmark._lt_ext() is not None)"
    ' || true

run() {
    local name="$1"; shift
    echo "=== SHAPE: $name ==="
    apptainer exec --nv --cleanenv \
        --bind /scratch/work:/work \
        /scratch/kernel.sif \
        bash -c "
          export TMPDIR=/work/.ext_build
          export TORCH_EXT_BUILD_DIR=/work/.ext_build
          mkdir -p \"\$TMPDIR\"
          python3 /work/benchmark.py $* " || true
    echo
}

run tiny             --batch-size 1   --seq-len 64
run default          --batch-size 8   --seq-len 128
run long_seq         --batch-size 8   --seq-len 1024
run large_batch      --batch-size 256 --seq-len 128
run default_padded   --batch-size 8   --seq-len 128 --padding-ratio 0.3
run default_causal   --batch-size 8   --seq-len 128 --causal
run causal_padded    --batch-size 8   --seq-len 128 --causal --padding-ratio 0.3
run long_seq_padded  --batch-size 8   --seq-len 1024 --padding-ratio 0.3
