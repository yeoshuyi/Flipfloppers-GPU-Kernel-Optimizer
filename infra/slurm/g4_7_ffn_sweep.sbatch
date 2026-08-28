#!/bin/bash
#SBATCH --job-name=g4_7_swp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

# G4.7 Stage 0a -- correctness gate for the FFN epilogue (cfg 51-76) added to
# csrc/g4_4_warpspec_gemm.cu. No timing here on purpose; step 37/41's rule is
# that a low-level kernel earns a throughput number only after a deterministic
# correctness gate, because fast-and-wrong is worse than no number.

set -uo pipefail

mkdir -p /scratch/work/.ext_build

apptainer exec --nv --cleanenv \
    --bind /scratch/work:/work \
    /scratch/kernel.sif \
    bash -c '
      export TMPDIR=/work/.ext_build
      export TORCH_EXT_BUILD_DIR=/work/.ext_build
      mkdir -p "$TMPDIR"
      python3 /work/experiments/g4_7_ffn_sweep.py
    '
