# RESUME — live cursor

## Task: optimize the Row-14 chunked causal forward for latency

`UserOptimizedTransformer._chunked_forward_causal` (`benchmark.py:1246`) works
and is committed (`b751393..659bb5b` on `main`, verified job 198
`OVERALL: PASS`). **This arc = make it fast.**

- **Baseline to beat:** **13.0 s / forward, 20.8 GB peak** (job 198,
  `results/logs/g7_0_chunked_oversize_run198.log`), Row 14 =
  `B=32 d=1024 H=16 hd=64 S=100000 L=2 ffn=1024` causal, `chunk_q=2048`
  → 49 chunks × 2 layers, K/V cache 12.21 GB fp16.
- **The answer is now PINNED** (G7.1, job 202):
  `experiments/g7_0_row14_golden.json` is the benchmark — the frozen harness
  cannot score row 14, so every step must reproduce this fingerprint under
  `abs<0.002 ∨ rel<0.02` (probe **check 6**). Reference = ONE forward over a
  FRESH input (`_chunked_forward_causal` mutates `x` in place and returns
  it). Re-baseline it ONLY deliberately, by re-running
  `infra/slurm/g7_1_gate_calibration.sbatch`.
- **Objective:** latency. **Hard gates:** correctness (`failed==0` on the
  disjunctive `abs<0.002 ∨ rel<0.02`) + peak ≤ ~22 GB on the 24 GB card.
- **Ideal floor:** attention is O(S²) ≈ 1.31e15 FLOPs ≈ 7.9 s at the
  165 TFLOP/s roofline; the 8 GEMMs ≈ 0.5 s. So ~4.5 s (≈1.5×) is
  loop / launch / merge / cast / copy overhead. Realistic target ≈ **8–10 s**.

Full plan: `~/.claude/plans/crispy-cooking-pine.md`. See [[resume_discipline]],
[[kernel_opt_loop_state]].

## Status

| phase | status |
|---|---|
| Row-14 chunking *capability* (prior arc) | ✅ DONE — `b751393..659bb5b`, job 198 PASS, job 199 regression 13/13 |
| **G7.1 gate in bytes + Row-14 golden** | ✅ DONE — `741f5fd`, `8604192`, jobs 200/201/202 calibrate+pin, **job 203 g7_0 OVERALL PASS**, **job 204 sweep 13/13 byte-identical to run168, no row slower** |
| Row-14 *optimization* (this arc) | **NOT STARTED** — start at Step 0 |

## Code anchors (`benchmark.py`)

| what | line |
|---|---|
| `_chunked_forward_causal` | 1246 |
| `prefix_causal_attn` (2× mem-efficient SDPA + fp32 LSE merge) | 1346 |
| `pre()` / `post()` position-wise halves | 1372 / 1377 |
| chunk loop body | 1414–1427 |
| `per_row` transient estimate (chunk_q sizer) | 1327 |
| separate final-norm streamed pass | 1430–1432 |
| `_CHUNK_*` env knobs | 262–266 |
| `_split_heads_view` (SHARED with rows 1–13 — do NOT edit) | 959 |
| `_would_oom_causal` (G7.1 byte gate) + `forward()` call site | 985 / ~1090 |
| `_causal_vram_budget` / `_causal_capture_bytes` (G7.1) | ~992 / ~1020 |

## How it works now

Per chunk: `pre` = `xc.float()`→`F.layer_norm`(folded, no affine)→`.to(fp16)`
→fused-QKV `F.linear`. Then split + `_split_heads_view` → write k,v into
persistent `kbuf/vbuf` `[B,H,S,hd]` fp16. Then `prefix_causal_attn`: calls
`torch.ops.aten._scaled_dot_product_efficient_attention` **twice** (square
causal diagonal block `[c0:c1]` + strictly-past non-causal block `[0:c0]`,
merged in fp32 by log-sum-exp) — chunk 0 is 1 call, ≈194 launches for Row 14.
Then `ctx.to(fp16)`; `post` = `ctx.transpose(1,2).contiguous().view` (the one
hot copy) → out_proj `F.linear` → residual → `h1.float()`→`F.layer_norm`
→`.to(fp16)` → ffn_in `F.linear` fp16 → `.float()` → `F.gelu` fp32 →
**ffn_out `F.linear` with ORIGINAL fp32 weights (load-bearing, do not touch)**
→ residual → `.to(fp16)`. Then a separate streamed pass applies `self.final_norm`.

## Roadmap (each step independently shipped + gated)

| # | step | expected | cum. 13.0 s → |
|---|---|---|---|
| 0 | **profile** — reuse `experiments/final_scorecard.py` `census()`+`bucket()` on `_chunked_forward_causal` for Row 14, `iters=1-3`, NVTX-split diagonal vs past SDPA. Decides where 1–7 pay off. No `benchmark.py` change. | — | 13.0 s |
| 1 | **A1** — in `prefix_causal_attn`, for `not wide`, swap the mem-efficient op → `torch.ops.aten._scaled_dot_product_flash_attention` (same `[B,H,S,hd]` layout, returns `(out, logsumexp[B,H,Lq], …)`). Keep both blocks + fp32 merge. `wide` stays mem-efficient. | −5…12% | ~11.8 s |
| 2 | **A2-probe** — add `check_flash_bottomright()` to the g7_0 probe: does raw `torch.ops.aten._flash_attention_forward(q_bshd, k_bshd, v_bshd, None, None, Lq, c1, 0.0, True, False, scale=1.0)` with `Lq<Lk` give **bottom-right** (= prefix-causal) in ONE call? + LSE shape + stride strictness + `._schema` dump. FAIL ⇒ skip A2. | — | — |
| 3 | **A2** (headline) — if probe passes: relayout K/V cache to `[B,S,H,hd]` contiguous (`_split_heads_bshd`, no transpose), ONE flash call per chunk, drop the 2nd call + LSE + fp32 merge + `out_*.float()` + the `.contiguous()` copy. `post()` takes `[B,L,d]` (`out.reshape` free for fp16). `wide` path unchanged (transpose to `[B,H,S,hd]` views). | −15…28% | ~9.5–10 s |
| 4 | **D1** fuse final-norm into last layer's chunk loop; **D2** drop the fp16→fp32→fp16 LN round-trips for `n1`/`n2` (fp16 path only — CUDA `layer_norm` already fp32-accumulates for half in, bit-identical). NOT `hidden.float()`/GELU/`ffn_out`/`wide`. | −2…6% | ~9.2–9.7 s |
| 5 | **B** retune `chunk_q` — re-derive `per_row` (`8d`→`4d` after D2; merge temps gone after A2), drop `_CHUNK_RESERVE_GB` 3.0→~2.0, adaptive sizer picks ~4096–5120. **OOM-sweep first** (`CHUNK_Q` env {2048,3072,4096,5120,6144} across Row14 + `(8,200000,1024)` + `(64,50000,1024)`, keep every peak ≤ 22 GB). | −4…10% | ~8.6–9.3 s |
| 6 | **C** compile the whole chunk body incl. the flash op — one `chunk_step(...)`, `torch.compile(dynamic=False)` + `torch._dynamo.mark_dynamic(k_prefix, 1)`. NOT reduce-overhead. Behind `_CHUNK_COMPILE`; flip default only if `recompiles==0` + clean win. | −3…10% | ~8.2–9 s |
| 7 | **E** (stretch) 2nd CUDA stream: overlap `pre(i+1)` with `attention(i)` (disjoint slice). High risk (races on `kbuf`/`x`). Ship only if clean ≥3% multi-seed win; else documented negative in `experiments/`. | −3…6% | ~7.9–8.7 s |

## Gates — run EVERY step

```bash
python3 tools/sync_entrypoint.py && python3 tools/verify_baseline.py
python3 tools/check_validity.py benchmark.py
sbatch infra/slurm/g7_0_chunked_oversize.sbatch      # OVERALL: PASS
#   check 1 prefix-causal vs full attn ; check 3 vs frozen baseline failed==0 ;
#   check 4 Row14 + (8,200000,1024) + (64,50000,1024): finite, peak <= ~22 GB, latency ;
#   check 5 Row14 B=4 fp16-store vs fp32-store: failed==0, max_abs not worse than ~8.1e-3 ;
#   check 6 Row14 B=32 vs experiments/g7_0_row14_golden.json: failed==0 AND
#           per-batch sum|y| drift <= 1e-3  <-- THE regression gate for this arc
sbatch infra/slurm/official_causal_sweep.sbatch      # 13/13 PASS, max_abs BYTE-IDENTICAL to
#   results/logs/official_causal_sweep_run168.log  (== run199.log == run204.log)
sbatch infra/slurm/g7_1_gate_calibration.sbatch      # only if the COMPILED path's
#   activation set changed -- re-validates the byte model + regenerates the golden
```

**Latency guard (rows 1-13 must not get slower).** The accuracy gate above is
not enough: diff per-row `optimized: median=` against
`results/logs/official_causal_sweep_run204.log` and flag any row regressing
>3%. Row 2 (0.0778 ms) is the sentinel for per-call CPU overhead — it is the
row that would catch an accidental driver call in `_would_oom_causal`. Row 6
(52.4841 ms) is the largest scored shape and the tightest gate margin (3.35x).

## Gate knobs changed by G7.1 (read before touching the gate)

`CHUNK_ACT_ELEMS` **changed meaning**: `0` (new default) = use the byte
model; `>0` = the OLD fixed `B*S*d >=` threshold, kept only so the probe can
force the gate on a small shape. New knobs: `CHUNK_OOM_FRAC` (0.80),
`CHUNK_BYTES_PER_D` (28), `CHUNK_BYTES_PER_FFN` (8), `CHUNK_BYTES_FIXED`
(128 MiB). The model predicts the **capture** pass (the warmed cudagraph
replay allocates ~nothing) and must **never under-predict** — there is no
OOM fallback. `g7_1_gate_calibration.sbatch` re-validates ≥1.25x headroom on
24 shapes; re-run it if the compiled path's activation set ever changes.

## Constraints

- `benchmark.py` edits **additive only** — no frozen-symbol lines
  (`verify_baseline` AST-checks 20 names). Regen `torch_transformer_benchmark.py`
  after every edit. `check_validity` bans `attn_mask=<not None>` (raw flash op
  has no such kwarg — fine), `data_ptr` outside `_mask_is_all_ones`,
  `register_buffer`/`nn.Parameter`.
- **`store=torch.float32` reference mode must keep working** (checks 3/5): no
  fp32 flash kernel on this stack → every flash change under `if not wide:`;
  `wide` stays mem-efficient + two-block split + fp32 LSE merge.
- **Accuracy margin is thin:** Row-14 fp16 vs fp32 chunked = `max_abs 8.1e-3`
  (over the 2e-3 atol), `mean_abs 3.4e-4`, passes only on the OR-rel branch.
  Don't spend it. fp32 GELU + fp32 `ffn_out` GEMM are load-bearing.
- Do NOT touch `_split_heads_view` (rows 1–13 share it) — A2 adds
  `_split_heads_bshd`.
- GPU = **sbatch only**, exclusive `rtx4090`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Why flash (the core lever)

`results/logs/g5_0_sdpa_backend_audit_run146.log` (job 146) + `docs/PROGRESS.md`
step 44: for fp16 causal, PyTorch `F.scaled_dot_product_attention` `auto`
**already picks FLASH** and flash is fastest-or-tied everywhere; mem-efficient
is 2.7% faster ONLY at row-6's huge token count, at a precision cost. The
chunked path is stuck on mem-efficient purely because it needs the returned
LSE for the two-block merge. A2 (if raw `_flash_attention_forward` is
bottom-right for `Lq<Lk`) removes the split entirely.

## Do-not-retry

- Hand-rolled fused CUDA attention — `csrc/g5_mega_causal.cu`, `docs/PROGRESS.md`
  step 49, ×0.74 vs cuBLAS+flash.
- EFFICIENT-vs-FLASH backend forcing for rows 1–13 — flash already optimal
  (step 44).
- Warp-spec / CUTLASS-grade GEMM — already 94–96% of the roofline (steps 45/49).
- Explicit banded `attn_mask` for prefix-causal — forces SDPA onto the slow
  MATH path (`docs/CATALOGUE.md`).

## Env knobs (`benchmark.py:262-266`)

`CHUNK_ACT_ELEMS` (0 = byte model; >0 = legacy override) · `CHUNK_OOM_FRAC`
(0.80) · `CHUNK_BYTES_PER_D` (28) · `CHUNK_BYTES_PER_FFN` (8) ·
`CHUNK_BYTES_FIXED` (128 MiB) · `CHUNK_MIN_SEQ` (2048) · `CHUNK_Q`
(0=adaptive) · `CHUNK_RESERVE_GB` (3.0) · `CHUNK_COMPILE` (0).
Add `CHUNK_FLASH` (A/B the flash path) if useful.

## Discipline

Commit continuously; update this file after every sbatch/commit with job IDs
+ expected `results/logs/*.log` + next action. User runs out of tokens
overnight — keep it a true cursor.
