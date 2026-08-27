# CLAUDE.md — Transformer Kernel Optimisation, RTX 4090 (sm_89)

Optimise `UserOptimizedTransformer` in `benchmark.py` against the frozen
`BaselineTransformer`.

**Load-on-demand — do NOT read these unless the trigger fires:**
- `docs/CAUSAL_LEDGER.md` — read FIRST when resuming causal-path work; fast
  table of every causal optimization shipped/dead/pending
- `docs/CATALOGUE.md` — read before proposing an optimisation
- `docs/DIAGNOSIS.md` — read after profiling, to map facts to actions
- `docs/MEGAKERNEL.md` — read only when working on G4
- `docs/SETUP.md` — infra, Phase 0 probe, measurement protocol (read once, day 1)
- `docs/AGENTS.md` — agent roles, limits, best practices (read once, at bootstrap)
- `/scratch/work/docs` and `/scratch/techjam2` - various docs here which were suppose to be combined (some outdated)

---

## INVARIANTS — violating any of these invalidates the run

1. **Never cache on `x.data_ptr()` for output. Never return a stale buffer.
   Never enqueue zero work.** The harness reuses one input tensor for 300+ timed
   calls. Exploiting that is not an optimisation.
2. **Accuracy failure ⇒ benchmark skipped ⇒ score 0.** Verify first, always. But YOU ARE AUTHORISED to use numerical rescue (Kahan summation, stochastic rounding) to fix near-miss accuracy failures.
3. **Never run `python` on the GPU directly. Always `sbatch`.** Direct runs
   bypass clock locking and corrupt every timing number.
4. **Fused weights go in PLAIN ATTRIBUTES**, never `Parameter`/`Buffer`.
   `load_state_dict(strict=True)` rejects new keys.
5. **After any "exact" transform, `max_abs` must be unchanged.** If it moved,
   the transform is wrong.
6. **Validate on the full sweep, not one shape.**

**Validity test for every proposal:** *would this still be correct if the input
distribution, the weights, and the shape all changed?* Yes → proceed. No but
gated on a runtime check with a correct fallback → acceptable, document the
gate. No → discard.

*AUTHORISED HIGH-RISK EXPLORATION:* SASS-level patching, Custom PTX `mma.sync` accumulators, ELF-header hex-editing to bypass `nvdisasm` bugs, and L2 cache/`cp.async` pipeline exploits.

---

## GROUND TRUTH

```
params 18,915,328    FP32 75.66MB  BF16 37.83MB  FP8 18.92MB
per layer FP8 3.15MB -> 24.1 KB/SM across 128 SMs
40.27 GFLOP/forward (B=8,S=128)

L2 72MB          <- BF16 model fits, FP32 does not (75.66 > 72)
shared 99 KB/SM  <- binding constraint for G4
```

**Floors (default shape):** TF32 0.487ms | BF16 0.244ms | FP8 0.122ms.
If a candidate beats the FP8 floor, check if it's computing the answer, but DO NOT dismiss it outright if using extreme pipeline fusion.

**Ada lacks:** TMA (use `cp.async`), `wgmma` (use `mma.sync.m16n8k16`), thread
block clusters.

---

## REGIME DISPATCH

The artefact is a dispatcher, not one kernel. **Name the regime in every proposal.**
All have causal and non causal
| Regime | Trigger | Bottleneck | Lever |
|---|---|---|---|
| TINY | `B·S < 128` | Launch | Graphs, min kernels, L2 pin, megakernel |
| DEFAULT | `128 ≤ B·S ≤ 16k` | Compute | FP8 FFN, fused QKV, fused FFN tile |
| LONG-SEQ | `S ≥ 1024` | Attention O(S²) | Flash mandatory, FP32 softmax accum |
| LARGE-BATCH | `B·S > 16k` | GEMM | TC occupancy, deep `cp.async` |
| PADDED | mask not all-ones | Masking | Modifier on the above |

```python
def forward(self, x, valid_token_mask=None):
    B, S, _ = x.shape; tok = B * S
    no_pad = self._mask_is_all_ones(valid_token_mask)
    if tok < 128:   return self._tiny(x, valid_token_mask, no_pad)
    if S >= 1024:   return self._long_seq(x, valid_token_mask, no_pad)
    if tok > 16384: return self._large_batch(x, valid_token_mask, no_pad)
    return self._default(x, valid_token_mask, no_pad)

def _mask_is_all_ones(self, m):
    if m is None: return True
    key = (m.data_ptr(), m.shape)   # caches a PROPERTY of the mask, not output
    hit = self._mask_cache.get(key)
    if hit is None:
        hit = bool(m.all()); self._mask_cache[key] = hit
    return hit
```

> The mask cache is the **one** legitimate `data_ptr()` use: it caches whether
> the mask is all-ones, never the result. The answer is still computed every call.

Thresholds are constants baked at solidification, not runtime-autotuned.

---

## PRECISION POLICY
**On accuracy failure, DO NOT IMMEDIATELY REVERT:** First, attempt numerical rescue. Try Kahan compensated summation, stochastic rounding, or block-microscaling before walking down the precision ladder.

---

## RUNNING THINGS

```python
from tools import submit, poll
res = poll(submit("cand_0042", "B8_S128", mode="accuracy"))
```

- **Always `sbatch`.** Never blocking `srun`, never direct `python`.
- Submit, release, poll. **Do not hold the GPU while reasoning.**
- Jobs are `--exclusive --gres=gpu:1`. Concurrent benchmarks produce noise.
- Clocks are the Slurm prolog's job. Never touch `nvidia-smi` clock state.

**Order per candidate — no exceptions:**
```
compile → tools/check_validity.py → run_accuracy (FULL sweep) → run_bench
```

---

## LOOP

```
1. Use the profiler subagent          (isolates ncu output from your context)
2. Read docs/DIAGNOSIS.md, map facts -> action
3. Read docs/CATALOGUE.md, pick ONE optimisation. Cite the profiler fact.
4. Implement one diff
5. python3 tools/check_validity.py    <- static gate, free
6. run_accuracy full sweep            <- FAIL = apply numerical rescue patch and retry.
7. run_bench
8. python3 tools/archive.py commit    <- script, no LLM
```

**Do not stop.** Push the boundaries. You are authorized to escalate immediately to PTX/SASS if standard torch/triton cannot express the necessary low-level hacks.

**Archive is 2D** (regime × family). An elite in one regime is not a global
winner. Never collapse onto the default shape.

**Solidification (final phase).** Strip all `triton.autotune`. Freeze the winning
constants per regime into static dispatch. Runtime branching consumes instruction
issue slots in exactly the TINY regime where they are scarcest.

---

## ACCURACY

```
external  max_abs 1e-3 OR max_rel 1e-2  (disjunctive, per element)
internal  max_abs 5e-4                  <- investigate above this
```

Reference is **already TF32** — you are not checked against exact FP32.
Track the *trend* in max_abs; 3e-4 → 8e-4 is a warning even though both pass.
Test `--padding-ratio 0.3` and `--causal`. **Check `S≥1024` every time you touch
precision** — accumulation drift is the most common silent failure.

---

## TOKEN DISCIPLINE

- **Never paste raw `ncu` output into context.** It is 25k–100k tokens. The
  profiler subagent exists solely to keep it out. Use it.
- Read files by range, not whole. Prefer `Grep` over `Read` when locating.
- One optimisation per iteration. Bundling makes failures undiagnosable and
  costs a re-run.
- Do not re-read `docs/*.md` you have already read this session.
- Report results as the JSON below and stop. No prose summaries of work already
  visible in the diff.

```json
{"improved": bool, "candidate_id": "str", "regime": "str",
 "applied": ["G0.1"], "speedup": 0.0, "accuracy_passed": bool,
 "max_abs_by_shape": {}, "fact_cited": "str", "next_hypothesis": "str"}
```

Do not end a turn with a question — there is no follow-up. State your assumption
and proceed.

---

## FOUR TRAPS (AND HOW TO BYPASS THEM)

1. `strict=True` rejects fused params → **plain attributes**
2. GeForce FP32-accumulate is **half rate** → 660 TFLOPS is FP16-accum only. **Bypass:** Use custom PTX with Kahan summation to fix the FP16-accum accuracy loss.
3. Explicit `attn_mask` kicks SDPA off flash → **`is_causal=True`**
4. Accuracy failure **skips the benchmark entirely** → **Bypass:** Correct the math, do not just abandon the kernel.

---

## OFFICIAL CAUSAL EVALUATION MATRIX

**Accuracy budget:** `atol=0.002`, `rtol=0.02` (disjunctive — see `compare_outputs()`). `benchmark.py --atol`/`--rtol` defaults match this.

**Official Test Shapes:**
| # | Batch Size | QKV Dim | Heads | Seq Len | Layers | Causal | FFN Dim |
|---|---|---|---|---|---|---|---|
| 1 | 64 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 2 | 1 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 3 | 4 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 4 | 16 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 5 | 128 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 6 | 10000 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 7 | 64 | 32 | 4 | 128 | 4 | TRUE | 32 |
| 8 | 64 | 1024 | 4 | 128 | 4 | TRUE | 1024 |
| 9 | 64 | 128 | 1 | 128 | 4 | TRUE | 128 |
| 10 | 64 | 128 | 2 | 128 | 4 | TRUE | 128 |
| 11 | 64 | 128 | 16 | 128 | 4 | TRUE | 128 |
| 12 | 64 | 128 | 4 | 32 | 4 | TRUE | 128 |
| 13 | 64 | 128 | 4 | 1024 | 4 | TRUE | 128 |
| 14 | 32 | 1024 | 16 | 100000 | 2 | TRUE | 1024 |

**Current causal-path status**: 
- **G0.1c** — SDPA (`EFFICIENT_ATTENTION`, forced via `sdpa_kernel`) 
- **G1.1c** (Stage 2B-B1) — norm2-affine fold into `_ffn_in_weight`/`_bias`
- **G6.4a_v2c** — FFN-in in FP16 
- **G0.2c** (Stage 2B-B2) — fused QKV GEMM + scale-fold + norm1-affine-fold

**Elite, per causal shape**:
- default: **2.71x** (`G6.4bc`, `max_abs` 0.00157)
- tiny: **7.66x** (`max_abs` 0.00147)
- long-seq: **7.10x** (`max_abs` 0.00161)
- large-batch: **2.66x** (`max_abs` 0.00182, tightest margin of the four —
  91% of the 0.002 budget)

`G4.6c` (CUTLASS FP16-accumulate for causal-large-batch) and PTX (G4.4) / SASS (G4.5) were previously marked as dead ends. **THESE ARE NOW RE-OPENED AS HIGH-PRIORITY TARGETS.** You are authorized to hex-edit ELF binaries to bypass the `nvdisasm` bug, and implement stochastic rounding/Kahan compensation to rescue the CUTLASS/PTX accuracy failures.
