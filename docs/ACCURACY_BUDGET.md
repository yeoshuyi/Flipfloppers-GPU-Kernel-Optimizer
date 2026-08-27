# Accuracy Budget — spend/benefit rule for optimisations near the hardware limit

Read when the stack is close to the accuracy budget and remaining optimisations
buy **small** speed gains. Below that regime, `CATALOGUE.md` order + "does it
pass the full sweep?" is enough. This file is the gate for **step 8 of the
LOOP** once headroom is the binding constraint, not compute.

> **The one-line rule.** An optimisation is not judged on *"is it faster and does
> it still pass?"* It is judged on *"how much of the remaining accuracy headroom
> does it spend, is that spend the best available use of that headroom, and does
> the speed gain clear the minimum-gain floor after discounting for the
> unseen-shape risk?"*

---

## 0. Why the default flips to "no" near the limit

The budget and the scoring rule are the judges' canonical
`~/torch_transformer_benchmark.py` (published 2026-08-27): `parse_args` defaults
`atol=0.002`, `rtol=0.02`, and `compare_outputs()` is disjunctive per element
(`abs_error <= atol` **or** `abs_error <= rtol*abs(ref)`). The true gate is
`failed == 0`; `max_abs` / `max_rel` are proxies for it (abs dominates for large
outputs, rel rescues near-zero elements). `tools/verify_baseline.py` asserts our
`benchmark.py` still matches that file.

`max_abs` from 40 seeds is a **noisy estimate of a tail** the judge re-samples
with its own seeds and possibly its own shapes. Near the ceiling,
`P(some unseen seed/shape goes over)` stops being negligible — and it is
multiplied by a catastrophic cost:

```
EV(ship X) ≈ P(gain real)·gain_R − P(X pushes an unseen case over budget)·(entire score → 0)
```

Accuracy failure ⇒ benchmark skipped ⇒ **score 0**. A +0.5% gain cannot outweigh
even a 2–3% chance of a zero. So near the limit the rational policy is
conservative: **bank headroom, remove marginal risk, spend budget only on large
gains.**

---

## 1. Ship ceiling and reserve

| line | value (of `atol=0.002`) | meaning |
|---|---|---|
| **hard ship ceiling** | `max_abs ≤ 0.00180` (90%) | never ship a stack above this on any swept shape |
| **target with reserve** | `max_abs ≤ 0.00170` (85%) | leave ≥ 0.0003 for a known high-value future optimisation |
| warning band | 0.00160–0.00180 | ship only with a rescue applied (§4) and a stated reason |

**Per-cell headroom** (like the 2D archive — budget is spent per `regime ×
causal`). Current shipped elites:

| cell | `max_abs` now | headroom to 0.00180 | status |
|---|---|---|---|
| causal-large-batch | **0.00182** | **−0.00002** | **FROZEN** — over the hard line. Lossy adds forbidden; buy-back only (§5). |
| causal-large-batch @ d512/ffn2048 | 0.00182 | −0.00002 | G4.7 (step 42) engaged here, `max_abs` **unchanged** — precision-neutral, exact ledger match |
| causal-long-seq | 0.00161 | 0.00019 | tight |
| causal-long-seq @ d512/ffn2048 | 0.00161 | 0.00019 | G4.7 engaged, `max_abs` unchanged |
| causal-default | 0.00157 | 0.00023 | tight |
| causal-tiny | 0.00147 | 0.00033 | some room |
| large-batch (non-causal) | 0.00158 | 0.00022 | tight (rose 0.00124→0.00158 for G4.3's +5.4%) |
| long-seq (non-causal) | 0.00137 | 0.00043 | room |
| default (non-causal) | 0.00113 | 0.00067 | room |
| tiny (non-causal) | 0.00084 | 0.00096 | room |

> Numbers from CLAUDE.md's causal ledger + `results/final_reverify_run118.log` +
> PROGRESS step 41. Refresh this table whenever the shipped stack changes.

---

## 2. Step 0 — classify: exact or lossy

Only **lossy** steps enter this analysis.

- **Exact** (`max_abs` unchanged to the last bit — INVARIANT 5): affine/weight
  folding (G1.x), fusion with matching math, layout/swizzle, CUDA graphs (G2.4),
  the G4.7 exact-erf GELU epilogue. **Cost zero budget — take if faster,
  always.** If `max_abs` moved, the transform is wrong, not "slightly lossy".
- **Lossy**: any precision reduction (FP16/BF16/FP8 storage or accumulate,
  skip-max softmax, polynomial activations, per-tensor quant). Goes to §3.

---

## 3. The metric

For a candidate lossy optimisation in regime `R`, measured on the **full shipped
stack + candidate**, 40 seeds, on the **binding shape of `R`** (found, not
assumed — large K → accumulation, large S → softmax drift, large batch → worse
tail order-statistic):

```
spend_R    = max_abs_R(stack + cand)  −  max_abs_R(stack)
headroom_R = 0.00180  −  max_abs_R(stack)                 # to hard ceiling; use 0.00170 if reserving
gain_R     = whole-model speedup in regime R, matched BEFORE/AFTER, one job, locked clocks
efficiency = gain_R  /  (spend_R / headroom_R)            # speed bought per fraction of remaining headroom
```

| condition | verdict |
|---|---|
| `spend_R ≤ noise` (≈ ±0.00003) | treat as exact; ship if `gain_R` > speed noise |
| `spend_R > headroom_R` | **reject** — or move down the rescue curve (§4) and re-measure |
| `0 < spend_R ≤ headroom_R` **and** `gain_R < 0.3%` | **reject** — below the minimum-gain floor (§6) |
| `0 < spend_R ≤ headroom_R` **and** `gain_R ≥ 0.3%` | rank by `efficiency` vs other candidates and vs reserve value (§5); ship only if it is the best marginal use of that headroom |
| cell is FROZEN (headroom ≤ 0) | **reject** unconditionally |

---

## 4. Every lossy optimisation is a (speed, budget) curve, not a point

Rescue moves you **along** the curve — a little speed back for a lot of budget:
FP32 softmax accumulation, Kahan / compensated summation, stochastic rounding,
split-FP32-accumulate (`SPLIT`), block microscaling.

Reference point from the stack: **G4.3** — `cfg26` (`max_abs` 0.00189, faster) →
`cfg48` = `cfg26` + `SPLIT 64` (`max_abs` 0.00125, ~0.5% model time). The ship
decision picked the curve point that maximised speed subject to
`max_abs ≤ ceiling − reserve`. A candidate that fails at `spend > headroom` is
not dead until its **whole rescue curve** is below the minimum-gain floor.

---

## 5. Reserve value, and re-evaluating what already shipped

Remaining headroom is a capital reserve. Spend it on candidate `X` only if `X`'s
efficiency beats **every other candidate's** — including known future
optimisations (FP8 FFN, deeper `cp.async`, …). Don't spend 0.0002 now on +0.5%
if a +5% optimisation needs 0.0003 later.

**Shipped optimisations are not sacred.** Each holds budget that may be better
spent. Maintain the ledger below and run a **Pareto check**: is there a shipped
lossy opt with tiny `Δspeed` and non-trivial `Δmax_abs` that can be reverted (or
rescued) to fund a much larger candidate? If so — revert, reallocate, net win at
equal budget.

**Buy-back for a FROZEN cell** (causal-large-batch): the 0.00182 is mostly FP16
attention *storage* (`G6.4bc`). Cheapest-accuracy-per-speed-lost first:
FP32 softmax accumulation only → stochastic rounding on the FP16 casts →
partial revert (FP16 QKV, FP32 score/PV matmul) → full revert. Each is a curve
point; measure them before declaring the cell unrecoverable.

---

## 6. Minimum-gain floor

Below **~0.3–0.5% whole-model** gain, do not ship a lossy optimisation
*regardless* of budget:

- another dispatch branch costs issue slots in exactly the TINY regime where
  they are scarcest (solidification note, CLAUDE.md);
- more code surface = more chance of a silent wrong answer on an unseen shape;
- the speed gain's relative measurement error is large at this size.

This floor also argues for **actively removing** marginal shipped lossy opts to
rebuild reserve, not just declining new ones.

---

## 7. Measurement rigor required for a marginal decision

| need | why |
|---|---|
| track `max_abs`, p99.9, **and** `failed`-count across seeds | one max is a noisy tail estimate |
| ≥ 40 seeds; more (or an adversarial-input search) within 15% of ceiling | tail shape matters at the margin |
| matched BEFORE/AFTER in one job, locked clocks; discount sub-1% gains for error bars | small gains have large relative noise |
| confirm the binding shape per optimisation | error concentrates at specific K / S / batch |
| re-measure on the **stacked** config, not the candidate alone | errors combine between RSS (√n) and linear (n); the tanh-vs-erf and "second FP16-accumulate source" cases were linear |

---

## 8. Ledger — shipped precision-affecting optimisations

Hard numbers where a matched BEFORE/AFTER exists this project's records;
`~approx` where only the stacked `max_abs` is on record and the per-step delta
is inferred from the step's own writeup. `Δmax_abs` is on the binding shape.

| id | cell(s) | exact/lossy | Δspeed (regime) | Δmax_abs (binding shape) | rescue | keep? |
|---|---|---|---|---|---|---|
| G6.4bc — FP16 attention storage | causal-* | lossy | large (step 28: biggest single-iter win) | ~approx +0.0006 → causal-large-batch **0.00182** (91%) | — | yes — but this is what FROZE causal-large-batch |
| G6.4a_v2c — FFN-in FP16 storage | causal-* | lossy | modest | ~approx, `max_abs` 0.00084–0.00100 non-causal (step 27 v2); revived only once budget was 0.002 | — | yes |
| G4.3 — warp-spec **FP16-accumulate** GEMM (attn) | large-batch, long-seq (**non-causal**) | lossy | **+5.4% / +4.75%** | **+0.00034** (lb 0.00124→0.00158) / **+0.00012** (ls 0.00125→0.00137) | SPLIT 64 (`cfg48`) — without it lb hits 0.00189 (99%) | yes |
| G4.3 for **causal** | causal-large-batch / long-seq | lossy | — | +0.0012 best-mitigated → **fails** (step 41) | every rung tried | **no** — closed |
| G4.7 — fused `ffn_in` + exact-erf GELU, **FP32-accumulate** | causal @ d512/ffn2048 (long_seq, large_batch) | **exact** | **+9.5% / +12.1%** | **0** — `max_abs` bit-identical to the pre-G4.7 40-trial record (0.00161 / 0.00182) | n/a — nothing to rescue | yes (step 42) |
| G4.7 — same, **FP16-accumulate** arm (`cfg51`) | — | lossy | x1.2–x2.2 on `ffn_in` | +2.1e-3 over 2 layers at K=512 (job 138) — over budget | — | **no** — not wired |

**Takeaway:** every lossy step in the causal stack landed *before* this session;
together they put causal-large-batch at 91% and froze it. The two additions
this session that actually shipped — G4.3 (non-causal) and G4.7 (causal) — were
chosen precisely because one had headroom in its regime and the other spends
**zero** budget.

**Candidate future optimisations** (rank by estimated efficiency; spend reserve
only on the top of this list):

| id | cell(s) | est. Δspeed | est. Δmax_abs | notes |
|---|---|---|---|---|
| FP8 e4m3 FFN | non-causal default/large-batch | ~1.5x FFN | needs probe | error averages down `eps/√K`; never in attention |
| G4.7 `ffn_out` GEMM + residual epilogue | causal @ d512/ffn2048 | step 41's other half | ~0 if FP32-accumulate | needs an fp32 residual add fused into the store |
| G4.7 FP16-accum arm on small-`ffn_dim` official rows | causal rows 1/6/13 | x1.2 on `ffn_in` | +~1e-3 | only if causal-large-batch headroom is bought back upstream first |

**Candidate future optimisations** (rank by estimated efficiency; spend reserve
only on the top of this list):

| id | cell(s) | est. Δspeed | est. Δmax_abs | notes |
|---|---|---|---|---|
| FP8 e4m3 FFN | non-causal default/large-batch | ~1.5x FFN | needs probe | error averages down `eps/√K`; never in attention |
| G4.3 into non-causal FFN GEMMs | long-seq, large-batch (non-causal) | x1.42–1.55 on ffn_in (run132) | fp16-accum: large; ACCF32: ~0 | ACCF32 arm is precision-neutral but half-rate — only wins at d512/ffn2048 |

---

## 9. Hooks

- **LOOP step 8** (`CLAUDE.md`): once any swept cell is > 85% of budget, the
  archive commit must carry `max_abs` and this file's decision table gates the
  ship, not `speedup > prev`.
- **`tools/archive.py`**: add `--max-abs` to `commit`; add a `headroom` command
  that prints the §1 per-cell table from the elite entries. Until then, keep §1
  updated by hand.
- **`tools/check_validity.py`**: static only; it cannot see `max_abs`. This file
  is the dynamic complement.
- **`tools/verify_baseline.py`**: run before every ship — confirms the budget
  (0.002/0.02) and the whole scoring harness still match the judges'
  `~/torch_transformer_benchmark.py`. If that file changes, every headroom
  number in §1 and every ledger row in §8 must be re-checked against the new
  budget.
