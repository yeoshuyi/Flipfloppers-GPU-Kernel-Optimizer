# RESUME — live cursor

## Current task: COMPLETE ✅

**Support official Row 14** (`B=32 d=1024 H=16 S=100000 L=2 ffn=1024 causal`)
— and any causal shape above the 24 GB activation limit — via a sequence-
**chunked** eager causal forward. Baseline stays OOM (frozen harness emits no
Row-14 number); the shipped model executes the shape and it is proven correct.

Plan: `/home/techjam2/.claude/plans/crispy-cooking-pine.md` (approved, done).

| phase | status |
|---|---|
| P1 `benchmark.py` `_CHUNK_*` + `_would_oom_causal` + `forward()` gate + `_chunked_forward_causal` | ✅ `b751393` |
| P2 `experiments/g7_0_chunked_oversize.py` + sbatch | ✅ `e6c6c21`, top-left fix `2b45c0c` |
| P3 `run_eval.sh` `RUN_ROW14=1` → probe | ✅ `4d30211` |
| P4 sbatch probe → real numbers | ✅ **job 198 OVERALL: PASS** → `results/logs/g7_0_chunked_oversize_run198.log` |
| P5 regression rows 1-13 | ✅ **job 199: 13/13 PASS**, `max_abs` byte-identical to run 168 → `results/logs/official_causal_sweep_run199.log` |
| P6 docs (README, FINAL_SCORECARD, PARETO, DEVPOST, ARCHITECTURE, PROGRESS 53) | ✅ `e9178b8` |
| P7 `make package` + `verify_submission.sh` | ✅ `dist/techjam2_e9178b8.tar.gz` → `verify_submission: PASS` |

## Results (job 198)

- **Row 14**: shipped chunked forward **13.0 s**, **peak 20.80 GB** / 24 GB
  card, output finite + right shape. adaptive chunk_q 2048, 49 chunks, KV
  cache 12.21 GB FP16.
- Also: `(8,200000,1024)` 12.1 s / 11.6 GB · `(64,50000,1024)` 7.5 s / 20.8 GB.
- **Row-14 accuracy** (B=4, FP16-store vs FP32-store chunked): `failed
  0 / 4.096e8`, mean_abs `3.4e-4`, max_abs `8.1e-3` — passes disjunctive
  `abs<0.002 ∨ rel<0.02`. Contingency (FP32 residual + block-flash) NOT needed.
- **CHUNK_COMPILE** A/B: 13.08 → 12.79 s (+2.2 %), default stays off.
- Equivalence (small): FP16 chunked vs frozen baseline `failed=0`; FP32
  chunked vs baseline max_abs `5.1e-4`. Gate never trips on the 13 official
  rows; auto-route bit-identical to a direct call.

## Regression (job 199)

13/13 PASS. `max_abs` byte-identical to run 168 on every row (0.0013676,
0.00195017, …). Speedups within run-to-run jitter (row 8, most stable, 1.933×
vs 1.932×). G7.0 gate does not fire; compiled path for rows 1-13 unchanged.

## Design notes (for future reference)

- **`SDPA(is_causal=True)` with `q_len ≠ kv_len` is TOP-LEFT aligned** in
  current PyTorch on every backend — NOT bottom-right. Job 197 check 1 caught
  this (MATH gave `max|part−full[c0:c1]| = 5.07`). The chunked path therefore
  splits each query chunk `[c0:c1]` → strictly-past non-causal block `[0:c0]`
  + square-causal diagonal block `[c0:c1]`, merged by log-sum-exp via
  `torch.ops.aten._scaled_dot_product_efficient_attention` (returns LSE). No
  `[chunk,c1]` mask is ever built.
- `_chunked_forward_causal` **mutates its input `x` in place** (residual
  stream) — only reachable behind the `_would_oom_causal` gate, never by the
  scored rows. `store=torch.float32` makes it its own accuracy reference.
- Env knobs: `CHUNK_ACT_ELEMS` (8e8) · `CHUNK_MIN_SEQ` (2048) · `CHUNK_Q`
  (0=adaptive) · `CHUNK_RESERVE_GB` (3.0) · `CHUNK_COMPILE` (0).
- `benchmark.py` diff across the whole arc is **purely additive** — no
  frozen-symbol lines touched (`verify_baseline` green).

## If more is wanted

- A **chunked baseline** so the harness can actually *score* Row 14 (currently
  it can't — reference OOMs first).
- Multi-GPU sharding to bring the 13 s single-card chunked forward down.
- `_chunked_forward_causal` currently no-pad only; padded oversize raises
  `NotImplementedError`.
