#!/bin/bash
#SBATCH --job-name=g4_3_ship
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

# G4.3 ship verification.
#   PASS 1 (BEFORE) = git HEAD's benchmark.py, copied verbatim to
#                     /work/_before_benchmark.py. 5 accuracy trials -- it is
#                     already verified (results/final_reverify_run118.log);
#                     this pass exists ONLY to produce a speedup measured in
#                     the SAME job, on the same locked clocks, as the after.
#   PASS 2 (AFTER)  = the working-tree benchmark.py with G4.3 wired in.
#                     40 accuracy trials, the project's shipping bar.
# Shapes: the two the G4.3 gate engages at (long_seq tok=8192, large_batch
# tok=32768, causal and non-causal, padded and not) plus tiny/default as
# controls that MUST come back unchanged (their token counts are below the
# gate, so G4.3 is not even registered there).

set -uo pipefail

run() {
    local tag="$1"; shift
    local script="$1"; shift
    local trials="$1"; shift
    echo "=== ${tag} ==="
    apptainer exec --nv --cleanenv --bind /scratch/work:/work /scratch/kernel.sif \
        python3 "$script" --accuracy-trials "$trials" --benchmark-on-failure "$@" || true
    echo
}

mkdir -p /scratch/work/.ext_build
export TMPDIR=/work/.ext_build
export TORCH_EXT_BUILD_DIR=/work/.ext_build

for pass in BEFORE AFTER; do
  if [ "$pass" = "BEFORE" ]; then S=/work/_before_benchmark.py; T=5; else S=/work/benchmark.py; T=40; fi
  echo "##################### PASS: $pass ($S, $T accuracy trials) #####################"
  run "$pass tiny"                     "$S" "$T" --batch-size 1   --seq-len 64
  run "$pass default"                  "$S" "$T" --batch-size 8   --seq-len 128
  run "$pass long_seq"                 "$S" "$T" --batch-size 8   --seq-len 1024
  run "$pass large_batch"              "$S" "$T" --batch-size 256 --seq-len 128
  run "$pass default_causal"           "$S" "$T" --batch-size 8   --seq-len 128  --causal
  run "$pass long_seq_causal"          "$S" "$T" --batch-size 8   --seq-len 1024 --causal
  run "$pass large_batch_causal"       "$S" "$T" --batch-size 256 --seq-len 128  --causal
  run "$pass large_batch_causal_pad"   "$S" "$T" --batch-size 256 --seq-len 128  --causal --padding-ratio 0.3
  run "$pass long_seq_causal_pad"      "$S" "$T" --batch-size 8   --seq-len 1024 --causal --padding-ratio 0.3
done
