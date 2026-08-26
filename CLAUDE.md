# CLAUDE.md — Transformer Kernel Optimisation, RTX 4090 (sm_89)

Optimise `UserOptimizedTransformer` in `benchmark.py` against the frozen
`BaselineTransformer`.

**Load-on-demand — do NOT read these unless the trigger fires:**
- `docs/CATALOGUE.md` — read before proposing an optimisation
- `docs/DIAGNOSIS.md` — read after profiling, to map facts to actions
- `docs/MEGAKERNEL.md` — read only when working on G4
- `docs/SETUP.md` — infra, Phase 0 probe, measurement protocol (read once, day 1)
- `docs/AGENTS.md` — agent roles, limits, best practices (read once, at bootstrap)

---

## INVARIANTS — violating any of these invalidates the run

1. **Never cache on `x.data_ptr()` for output. Never return a stale buffer.
   Never enqueue zero work.** The harness reuses one input tensor for 300+ timed
   calls. Exploiting that is not an optimisation.
2. **Accuracy failure ⇒ benchmark skipped ⇒ score 0.** Verify first, always.
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

Forbidden regardless of speedup: windowed/sparse attention, low-rank weights,
layer dropping, weight-dependent shortcuts.

---

## GROUND TRUTH

```
params 18,915,328    FP32 75.66MB  BF16 37.83MB  FP8 18.92MB
per layer FP8 3.15MB -> 24.1 KB/SM across 128 SMs
40.27 GFLOP/forward (B=8,S=128)

L2 72MB          <- BF16 model fits, FP32 does not (75.66 > 72)
shared 99 KB/SM  <- binding constraint for G4
```

**Precision ladder — GeForce Ada runs FP32 accumulate at HALF RATE:**
```
TF32  82.6 TFLOPS  <- baseline runs here
BF16 165.2
FP8  330.3         <- cuBLASLt ceiling (requires COMPUTE_32F)
FP8  660.6 w/ FP16 acc -- mma.sync PTX only. NEVER cite as available.
```

**Floors (default shape):** TF32 0.487ms | BF16 0.244ms | FP8 0.122ms.
A candidate beating the FP8 floor is not computing the answer.

**Ada lacks:** TMA (use `cp.async`), `wgmma` (use `mma.sync.m16n8k16`), thread
block clusters.

---

## REGIME DISPATCH

The artefact is a dispatcher, not one kernel. **Name the regime in every proposal.**

| Regime | Trigger | Bottleneck | Lever |
|---|---|---|---|
| TINY | `B·S < 128` | Launch | Graphs, min kernels, L2 pin, megakernel |
| DEFAULT | `128 ≤ B·S ≤ 16k` | Compute | FP8 FFN, fused QKV, fused FFN tile |
| LONG-SEQ | `S ≥ 1024` | Attention O(S²) | Flash mandatory, FP32 softmax accum |
| LARGE-BATCH | `B·S > 16k` | GEMM | TC occupancy, deep `cp.async` |
| PADDED | mask not all-ones | Masking | Modifier on the above |
| CAUSAL | `config.causal` | — | Compile-time constant, never per-layer branch |

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

```
FFN + out_proj    FP8 e4m3, PER-CHANNEL scales   <- 65% of FLOPs, K=2048
QKV, scores       BF16
softmax, LayerNorm FP32   <- never quantise
residual stream   FP32    <- cast down only at GEMM inputs
output            FP32    <- cast back before returning
```

FP8 survives 3 mantissa bits because error averages down over the reduction:
`eps/sqrt(K) = 6%/45.3 ~ 0.14%` against a 1% budget. Per-channel scaling removes
the systematic part. **Never FP8 in attention** — softmax tails die.

**On accuracy failure, walk down:** FP8 FFN+BF16 attn → FP8 FFN only →
split-precision (`A = A_hi + A_lo`, 3 BF16 matmuls) → BF16 everywhere.
**Use split-precision before abandoning low precision.**

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
6. run_accuracy full sweep            <- FAIL = score 0, diagnose, stop
7. run_bench
8. python3 tools/archive.py commit    <- script, no LLM
```

**Stop when** 3 consecutive iterations yield <2%, or budget expires.

**Escalate implementation** `torch → triton → cuda → ptx` only when the current
level demonstrably cannot express the strategy. Inductor may already produce the
fusion — profile before hand-writing.

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

## FOUR TRAPS

1. `strict=True` rejects fused params → **plain attributes**
2. GeForce FP32-accumulate is **half rate** → 660 TFLOPS is FP16-accum only
3. Explicit `attn_mask` kicks SDPA off flash → **`is_causal=True`**
4. Accuracy failure **skips the benchmark entirely** → correctness is a gate
