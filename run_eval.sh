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
# For every shape it prints, per batch, the baseline vs optimized timing and the
# numerical diff (max abs / max rel error, failed-element count) against the
# reference, then an aggregate + geometric-mean speedup.
#
# benchmark.py was removed (commit below): torch_transformer_benchmark.py is
# now the single source of truth, not a generated copy of one.
# Row 14 (seq_len=100000) is skipped by default -- the FP32 baseline OOMs a
# 24 GB card before any math (a single [32,100000,1024] fp32 activation is
# 12.2 GiB). Set RUN_ROW14=1 to attempt it anyway.
#
#   ./run_eval.sh                 # rows 1-13
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
# Row 14 is NEVER added to ROWS: the frozen harness cannot score it (the FP32
# reference OOMs a 24 GB card in generate_random_case / baseline's [B,H,S,S]
# scores before our model runs, and run_accuracy_tests has no try/except). With
# RUN_ROW14=1 we instead run experiments/g7_0_chunked_oversize.py -- the shipped
# model executes S=100000 via sequence chunking (torch_transformer_benchmark.py
# _chunked_forward_causal) and the probe proves it correct against a
# higher-precision reference. See docs/FINAL_SCORECARD.md.
if [[ "${RUN_ROW14:-0}" != "1" ]]; then
  echo "run_eval: skipping row 14 (frozen harness OOMs before our model); set RUN_ROW14=1 to run the chunked-capability probe"
fi

flag() { grep -oE -- "$1 +[0-9]+" <<<"$2" | grep -oE '[0-9]+' | head -1; }

SUMMARY="$OUTDIR/run_eval_${STAMP}_summary.txt"
HDR=$(printf '%-4s %6s %6s %5s %4s %11s %11s %9s %11s %12s %14s  %s' \
      row B S d H base_med_ms opt_med_ms speedup max_abs max_rel "failed(n/tot)" result)
{ echo "entry=$ENTRY  image=$SIF  $(date -u +%FT%TZ)"; echo; echo "$HDR"
  printf '%s\n' "$(printf '%.0s-' $(seq 1 ${#HDR}))"; } | tee "$SUMMARY"

TB=""; TO=""; SP=""; NFAIL=0
for spec in "${ROWS[@]}"; do
  n="${spec%% *}"; args="${spec#* }"
  B=$(flag '--batch-size' "$args"); S=$(flag '--seq-len' "$args")
  d=$(flag '--d-model' "$args");    H=$(flag '--heads' "$args")
  log="$OUTDIR/run_eval_${STAMP}_row${n}.log"
  echo ">>> row $n : B=$B S=$S d=$d H=$H"
  apptainer exec --nv --cleanenv --bind "$ROOT":/work "$SIF" \
      python3 "/work/$ENTRY" --causal $args 2>&1 | tee "$log" || true

  # --- parse timings + diff from the benchmark's own output ---------------
  bmed=$(grep -oE 'baseline *: median=[0-9.]+ ms'  "$log" | grep -oE '[0-9.]+' | tail -1)
  omed=$(grep -oE 'optimized: median=[0-9.]+ ms'   "$log" | grep -oE '[0-9.]+' | tail -1)
  spd=$( grep -oE 'speedup +: [0-9.]+x'            "$log" | grep -oE '[0-9.]+' | tail -1)
  sline=$(grep -E '^summary: (PASS|FAIL)'          "$log" | tail -1)
  res=$(  grep -oE 'summary: (PASS|FAIL)'          <<<"$sline" | awk '{print $2}')
  mabs=$( grep -oE 'max_abs=[0-9.eE+-]+'           <<<"$sline" | cut -d= -f2)
  mrel=$( grep -oE 'max_rel=[0-9.eE+-]+'           <<<"$sline" | cut -d= -f2)
  fail=$( grep -oE 'failed=[0-9]+/[0-9]+'          <<<"$sline" | cut -d= -f2)
  [[ -z "$res"  ]] && res="ERR/OOM"
  [[ "$res" == "FAIL" || "$res" == "ERR/OOM" ]] && NFAIL=$((NFAIL+1))

  printf '%-4s %6s %6s %5s %4s %11s %11s %8sx %11s %12s %14s  %s\n' \
    "$n" "$B" "$S" "$d" "$H" "${bmed:-NA}" "${omed:-NA}" "${spd:-NA}" \
    "${mabs:-NA}" "${mrel:-NA}" "${fail:-NA}" "$res" | tee -a "$SUMMARY"

  [[ -n "${bmed:-}" && -n "${omed:-}" ]] && { TB+="$bmed "; TO+="$omed "; }
  [[ -n "${spd:-}" ]] && SP+="$spd "
done

# --- aggregate ---------------------------------------------------------------
{ printf '%s\n' "$(printf '%.0s-' $(seq 1 ${#HDR}))"
  awk -v tb="$TB" -v to="$TO" -v sp="$SP" -v nf="$NFAIL" 'BEGIN{
    n=split(tb,B," "); split(to,O," "); m=split(sp,S," ")
    for(i=1;i<=n;i++){sb+=B[i]; so+=O[i]}
    for(i=1;i<=m;i++){if(S[i]>0){lg+=log(S[i]); k++}}
    printf "TOTAL  base %.3f ms  ->  opt %.3f ms   aggregate %.2fx   geomean %.2fx  (n=%d)\n",
           sb, so, (so>0?sb/so:0), (k>0?exp(lg/k):0), k
    printf "GATE   %s   (%d shape(s) failed / errored; the official gate is failed==0 on every shape)\n",
           (nf==0 ? "PASS" : "FAIL"), nf
  }'
} | tee -a "$SUMMARY"

# --- row 14: chunked-capability probe (never scored by the harness) ---------
if [[ "${RUN_ROW14:-0}" == "1" ]]; then
  r14log="$OUTDIR/run_eval_${STAMP}_row14_probe.log"
  echo >> "$SUMMARY"
  echo ">>> row 14 : chunked-capability probe (experiments/g7_0_chunked_oversize.py)"
  apptainer exec --nv --cleanenv --bind "$ROOT":/work \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$SIF" \
      python3 /work/experiments/g7_0_chunked_oversize.py 2>&1 | tee "$r14log" || true
  sline=$(grep -E '^ROW14_SUMMARY ' "$r14log" | tail -1)
  ov=$(grep -E '^OVERALL: ' "$r14log" | tail -1 | awk '{print $2}')
  get() { grep -oE "$1=[^ ]+" <<<"$sline" | cut -d= -f2; }
  sms=$(get shipped_ms); spk=$(get peak_gb)
  sab=$(get acc_max_abs); saf=$(get acc_failed); sb=$(get acc_b)
  {
    printf '%-4s %6s %6s %5s %4s  %s\n' 14 32 100000 1024 16 \
      "baseline: OOM (FP32 [B,H,S,S] scores)  |  shipped: ${sms:-NA} ms chunked, peak ${spk:-NA} GB  |  acc(B${sb:-?} fp16-vs-fp32): max_abs ${sab:-NA}, failed ${saf:-NA}  |  ${ov:-ERR} -> supported via sequence chunking"
  } | tee -a "$SUMMARY"
  [[ "$ov" == "PASS" ]] || echo "run_eval: WARNING -- row 14 probe did not report OVERALL: PASS (see $r14log)"
fi

echo
echo "run_eval: per-row logs + this summary in $OUTDIR  (stamp $STAMP)"
echo "run_eval: reference numbers -> docs/FINAL_SCORECARD.md"
[[ "$NFAIL" -eq 0 ]] || exit 1
