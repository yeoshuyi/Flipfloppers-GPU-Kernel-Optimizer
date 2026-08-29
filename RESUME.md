# RESUME — live cursor

**Task:** support official **Row 14** (`B=32 d=1024 H=16 S=100000 L=2 ffn=1024
causal`) — and any causal shape whose activation set exceeds the 24 GB card —
via a sequence-**chunked** eager causal forward. Baseline stays OOM (the frozen
harness emits no number for row 14); the shipped model must *execute* the shape
and we must *prove it correct* with a standalone probe.

Plan: `/home/techjam2/.claude/plans/crispy-cooking-pine.md` (approved).

## State

| phase | status |
|---|---|
| P1 `benchmark.py`: `_CHUNK_*` consts + `_would_oom_causal` + `forward()` gate + `_chunked_forward_causal` | DONE — commit `b751393` |
| P2 `experiments/g7_0_chunked_oversize.py` + `infra/slurm/g7_0_chunked_oversize.sbatch` | DONE — commit pending |
| P3 `run_eval.sh` `RUN_ROW14=1` → run the probe | DONE — commit `4d30211` |
| P4 sbatch the probe → real numbers | **DONE — job 198 OVERALL: PASS**, `results/logs/g7_0_chunked_oversize_run198.log` |
| P5 regression: rows 1-13 vs `official_causal_sweep_run168.log` (gate must NOT fire) | **IN FLIGHT — job 199** |
| P6 docs: README row-14 lines + `docs/{FINAL_SCORECARD,PARETO_FRONTIER_ANALYSIS,DEVPOST,ARCHITECTURE,PROGRESS}.md` (step 53) | in progress |
| P7 `make package` + `bash infra/verify_submission.sh` | not started |

## Job 198 results (all PASS) — `results/logs/g7_0_chunked_oversize_run198.log`

- **Row 14** (B=32 S=100000 d=1024 H=16 L=2 ffn=1024 causal): shipped chunked
  forward **13018 ms**, **peak 20.80 GB** / 24 GB card, output finite + right
  shape. adaptive chunk_q=2048, 49 chunks, KV cache 12.21 GB.
- Also ran (B=8 S=200000): 12055 ms, 11.64 GB. (B=64 S=50000): 7546 ms, 20.80 GB.
- **CHUNK_COMPILE** A/B on Row 14: 13079 → 12793 ms (**+2.2%**), default stays off.
- **Row-14 accuracy** (B=4, fp16-store vs fp32-store chunked): max_abs 8.13e-3,
  **mean_abs 3.41e-4**, **failed 0 / 409,600,000** — passes the disjunctive
  `abs<0.002 OR rel<0.02` gate. Contingency (fp32 residual + block-flash) NOT
  needed.
- Equivalence (small): fp16 chunked vs frozen baseline `failed=0`; fp32 chunked
  vs baseline max_abs 5.10e-4. Gate never trips on the 13 official rows;
  auto-route bit-identical to a direct `_chunked_forward_causal` call.
- Fix along the way (commit `2b45c0c`): `SDPA(is_causal=True)` with q_len≠kv_len
  is **top-left** aligned (every backend), so per-chunk attention is a
  past-block + square-causal-block split merged by LSE via
  `torch.ops.aten._scaled_dot_product_efficient_attention`.

## In flight

- **Slurm job 199** `official_causal_sweep.sbatch` → regression, rows 1-13.
  Expect speedups + max_abs to match `results/logs/official_causal_sweep_run168.log`
  (the G7.0 gate must NOT fire for any of them).

## Design (locked)

- **Gate** (`_would_oom_causal`, staticmethod): `B*S*d >= _CHUNK_ACT_ELEMS`
  (default `8e8` ≈ 20 GB; largest official row 1-13 is row 6 = `1.64e8`, ~5×
  under). Plus `x.is_cuda`, `no_pad`, `self.config.causal`, `S >= _CHUNK_MIN_SEQ`
  (2048); if over-budget but `S < 2048` → `NotImplementedError` (S-chunking
  can't help).
- **`_chunked_forward_causal`**: eager, no-pad only. Residual `x` kept in
  `store` (fp16 default) and **mutated in place**. K/V buffer `[B,H,S,hd]` in
  `store`, allocated once, refilled per layer. Per query chunk: LN→fused QKV
  (folded fp16 weights)→write K/V slice→`SDPA(q, kbuf[:, :, :c1], vbuf[:, :, :c1],
  is_causal=True, scale=1.0)` (bottom-right causal = exact prefix)→out_proj→
  residual→LN2→ffn_in→GELU(fp32)→ffn_out→residual. Final norm chunked in place.
  SDPA forced to FLASH/EFFICIENT (never math → never a `[B,H,c,S]` tile).
- **Optimization**: adaptive `chunk_q` from `mem_get_info()` (K/V bytes +
  `_CHUNK_RESERVE_GB` held back); K/V buffers reused across layers; per-chunk
  transients `del`'d; optional `_CHUNK_COMPILE=1` wraps the position-wise
  pre/post halves in `torch.compile` (2 shape variants) — probe reports the
  naive-vs-compiled delta.
- **Accuracy reference**: same method with `store=torch.float32` (residual +
  K/V + weights widened; SDPA forced EFFICIENT which supports fp32). fp64 is
  out (no fp32+ flash/efficient at S=100000 → math OOM).

## Env knobs (benchmark.py module consts, all `os.environ`)

`CHUNK_ACT_ELEMS` (8e8) · `CHUNK_MIN_SEQ` (2048) · `CHUNK_Q` (0=adaptive) ·
`CHUNK_RESERVE_GB` (3.0) · `CHUNK_COMPILE` (0)

## Verify (per plan)

```
python3 -c "import ast; ast.parse(open('benchmark.py').read())"
python3 tools/sync_entrypoint.py && python3 tools/verify_baseline.py
python3 tools/check_validity.py benchmark.py
sbatch infra/slurm/g7_0_chunked_oversize.sbatch
sbatch infra/slurm/official_causal_sweep.sbatch      # regression, rows 1-13
make package && bash infra/verify_submission.sh dist/techjam2_*.tar.gz
```

## Commits (RESUME updated after each)

1. `benchmark.py: _chunked_forward_causal for row-14-class causal shapes` (+ regen entrypoint)
2. `experiments: g7_0 row-14 chunked capability + accuracy probe` (+ sbatch)
3. `run_eval.sh: RUN_ROW14 runs the chunked probe`
4. (after job) `docs: Row 14 supported via sequence chunking — PROGRESS 53; ...`

## Contingency

If the probe's row-14 accuracy check reports `failed != 0` (fp16 residual over
the 0.002 budget at L=2): STOP, report. Fix path = fp32 residual + hand-rolled
block-flash (fp32 x 13.1 GB + fp16 K/V 13.1 GB won't co-fit → needs the
online-softmax merge from `csrc/g5_mega_causal.cu`, a bigger change). Needs a
user call — do not ship a lossy row-14 path silently.
