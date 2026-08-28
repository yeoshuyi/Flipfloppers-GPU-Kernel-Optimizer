#!/usr/bin/env bash
# Standardized evaluation entry point.
#
# Runs the submission over the official 14-row causal evaluation matrix
# (CLAUDE.md -> "OFFICIAL CAUSAL EVALUATION MATRIX") inside the Apptainer image,
# via the same path a grader would use:
#
#     apptainer exec --nv --bind $REPO:/work /scratch/kernel.sif \
#         python3 /work/torch_transformer_benchmark.py --causal <shape args>
#
# Override the entry point to the human-edited file with  ENTRY=benchmark.py.
# Row 14 (seq_len=100000) is skipped by default -- the FP32 baseline OOMs a
# 24 GB card before any math (a single [32,100000,1024] fp32 activation is
# 12.2 GiB). Set RUN_ROW14=1 to attempt it anyway.
#
#   ./run_eval.sh                 # rows 1-13
#   ENTRY=benchmark.py ./run_eval.sh
#   RUN_ROW14=1 ./run_eval.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${KERNEL_SIF:-/scratch/kernel.sif}"
ENTRY="${ENTRY:-torch_transformer_benchmark.py}"
OUTDIR="${OUTDIR:-$ROOT/results/logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# --- preflight -------------------------------------------------------------
[[ -e "$SIF" ]]            || { echo "run_eval: image not found: $SIF (build: bash infra/apptainer/build.sh)" >&2; exit 1; }
[[ -f "$ROOT/$ENTRY" ]]   || { echo "run_eval: entry point not found: $ROOT/$ENTRY" >&2; exit 1; }
[[ -f "$ROOT/csrc/g4_4_warpspec_gemm.cu" && -f "$ROOT/csrc/cublaslt_algo.cpp" ]] \
    || { echo "run_eval: csrc/ sources missing -- custom kernels will silently disengage" >&2; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "run_eval: warning -- nvidia-smi unavailable"
if [[ "$ENTRY" == "torch_transformer_benchmark.py" ]]; then
  python3 "$ROOT/tools/sync_entrypoint.py" --check \
    || { echo "run_eval: torch_transformer_benchmark.py is stale -- run tools/sync_entrypoint.py" >&2; exit 1; }
fi
mkdir -p "$OUTDIR"

# --- official matrix (verbatim from infra/slurm/official_causal_sweep.sbatch) --
ROWS=(
  "1  --batch-size 64    --d-model 128  --heads 4  --seq-len 128  --layers 4 --ffn-dim 128"
  "2  --batch-size 1     --d-model 128  --heads 4  --seq-len 128  --layers 4 --ffn-dim 128"
  "3  --batch-size 4     --d-model 128  --heads 4  --seq-len 128  --layers 4 --ffn-dim 128"
  "4  --batch-size 16    --d-model 128  --heads 4  --seq-len 128  --layers 4 --ffn-dim 128"
  "5  --batch-size 128   --d-model 128  --heads 4  --seq-len 128  --layers 4 --ffn-dim 128"
  "6  --batch-size 10000 --d-model 128  --heads 4  --seq-len 128  --layers 4 --ffn-dim 128"
  "7  --batch-size 64    --d-model 32   --heads 4  --seq-len 128  --layers 4 --ffn-dim 32"
  "8  --batch-size 64    --d-model 1024 --heads 4  --seq-len 128  --layers 4 --ffn-dim 1024"
  "9  --batch-size 64    --d-model 128  --heads 1  --seq-len 128  --layers 4 --ffn-dim 128"
  "10 --batch-size 64    --d-model 128  --heads 2  --seq-len 128  --layers 4 --ffn-dim 128"
  "11 --batch-size 64    --d-model 128  --heads 16 --seq-len 128  --layers 4 --ffn-dim 128"
  "12 --batch-size 64    --d-model 128  --heads 4  --seq-len 32   --layers 4 --ffn-dim 128"
  "13 --batch-size 64    --d-model 128  --heads 4  --seq-len 1024 --layers 4 --ffn-dim 128"
)
if [[ "${RUN_ROW14:-0}" == "1" ]]; then
  ROWS+=("14 --batch-size 32 --d-model 1024 --heads 16 --seq-len 100000 --layers 2 --ffn-dim 1024")
else
  echo "run_eval: skipping row 14 (baseline OOMs a 24 GB card); set RUN_ROW14=1 to attempt it"
fi

SUMMARY="$OUTDIR/run_eval_${STAMP}_summary.txt"
echo "entry=$ENTRY  image=$SIF  $(date -u +%FT%TZ)" | tee "$SUMMARY"
echo "row | accuracy | speedup" | tee -a "$SUMMARY"

for spec in "${ROWS[@]}"; do
  n="${spec%% *}"; args="${spec#* }"
  log="$OUTDIR/run_eval_${STAMP}_row${n}.log"
  echo ">>> row $n : $args"
  apptainer exec --nv --cleanenv --bind "$ROOT":/work "$SIF" \
      python3 "/work/$ENTRY" --causal $args 2>&1 | tee "$log" || true
  acc="$(grep -oE 'summary: (PASS|FAIL)[^|]*\| max_abs=[0-9.e-]+[^|]*\| failed=[0-9/]+' "$log" | tail -1)"
  spd="$(grep -oE 'speedup +: [0-9.]+x[^$]*' "$log" | tail -1)"
  printf '%-3s | %s | %s\n' "$n" "${acc:-?}" "${spd:-?}" | tee -a "$SUMMARY"
done

echo
echo "run_eval: per-row logs + summary in $OUTDIR (stamp $STAMP)"
echo "run_eval: reference numbers -> docs/FINAL_SCORECARD.md"
