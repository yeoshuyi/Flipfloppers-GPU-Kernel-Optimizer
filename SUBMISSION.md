# Optimizing a Frozen-Baseline Transformer on an RTX 4090: An Agentic Engineering Log

## Overview

We set out to optimize `UserOptimizedTransformer`, a 19M-parameter
transformer, against a frozen, unmodifiable `BaselineTransformer` — the exact
PyTorch reference implementation given to us, eager-mode, manual attention
math, no precision tricks. The rules were strict: match its output within a
tight numerical tolerance on every input shape, or the benchmark scores zero
regardless of speed. We had one consumer GPU, no datacenter hardware, and a
finite AI token budget to spend directing the work. This document covers what
we measured before and after, the constraints that shaped every decision we
made, how we structured Claude Code as an agentic collaborator under those
constraints, and what we shipped versus what we tried and rejected — and why
the rejections matter as much as the wins.

## Before / After

Every row compares our final optimized model directly against the original,
untouched `BaselineTransformer` — the actual starting point given to us, not
an intermediate version of our own work. Numbers below are from a fresh
end-of-project re-verification sweep (all 8 shapes, 40-seed accuracy rigor),
baseline and optimized measured together in the same run, on the same GPU.
(An earlier draft of this table showed smaller speedups; this table
supersedes it now that the FP16 FFN/attention passes and the causal-path
rewrite described later in this document have shipped.)

**Provenance:** `jobs/final_reverify.sbatch`, output preserved at
`results/final_reverify_run118.log` (previously only in
`/scratch/techjam2/runs/118.out`, outside the git repo and at risk of being
cleaned up — copied in so this table stays verifiable). These are
intentionally a fresh, single-job, all-shapes-together measurement rather
than a reassembly of the per-optimization-step numbers logged incrementally
in `archive/*.json` during development — small deltas against
`docs/CAUSAL_LEDGER.md`'s per-step numbers (e.g. causal 2.76x here vs
2.71x archived at `G6.4bc`'s own commit) are expected run-to-run/thermal
variance between separately-timed runs, not a discrepancy in what shipped;
both were verified against the same commit's `benchmark.py`.

| Regime | Baseline (`BaselineTransformer`) | Ours (`UserOptimizedTransformer`) | Speedup |
|---|---|---|---|
| Tiny (B·S < 128) | 1.452 ms | **0.201 ms** | **7.24x** |
| Default | 1.405 ms | **0.558 ms** | **2.52x** |
| Long-sequence (S ≥ 1024) | 24.713 ms | **4.600 ms** | **5.37x** |
| Large-batch | 41.908 ms | **17.083 ms** | **2.45x** |
| Padded | 1.408 ms | **0.564 ms** | **2.50x** |
| Causal | 1.547 ms | **0.561 ms** | **2.76x** |

Every one of these numbers required a full accuracy pass first — correctness
is a hard gate in this project, not a footnote. At the long-sequence shape in
particular, a 24.7 ms forward pass drops to 4.6 ms; that's not a rounding
trick, it comes from a real, verified change in which GPU kernels run at all
(more below). Causal moved the most late in the project — 1.78x at the
initial baseline-compile-only stage to 2.76x once SDPA, the algebraic folds,
and FP16 attention were independently re-verified and shipped for that path
too (`docs/CAUSAL_LEDGER.md` has the full per-step record).

**Since this sweep**, one further precision-neutral optimisation shipped for
the causal path (`docs/PROGRESS.md` step 42): a fused feed-forward GEMM +
exact-GELU epilogue on the warp-specialised kernel, lifting causal
long-sequence from 7.10x → **7.78x** and causal large-batch from 2.66x →
**2.98x** at the d512/ffn2048 shape, with the model's output bit-identical
to before. These two causal sub-shapes are not broken out in the table above
(whose "Long-sequence" / "Large-batch" rows are the non-causal d512 shapes);
`archive/causal-*__fp16.json` and `results/g4_7_ship_verify_v2_run142.log`
carry the numbers.

## Working Under Constraints

The two constraints that shaped nearly every technical decision in this
project were not the model or the benchmark — they were the hardware we had
and the AI budget we had to spend directing it.

**Consumer GPU, not datacenter silicon.** We were optimizing for an RTX
4090 (Ada Lovelace, sm_89) — the gaming/workstation tier, not an H100 or
B200. That closes off entire categories of technique that the literature on
transformer inference optimization takes for granted. Hopper's TMA (async
bulk memory copy) and `wgmma` (warp-group matrix multiply) don't exist on
Ada; asynchronous memory movement has to be hand-rolled with `cp.async` and a
software pipeline. Ada's FP8 tensor cores are real, but they're capped at
`CUBLAS_COMPUTE_32F` accumulation (330 TFLOPS) — the 660+ TFLOPS FP16-accumulate
tier that makes FP8 attractive on Hopper needs hand-written `mma.sync` PTX on
Ada, because cuBLAS simply cannot reach it on this chip. And GeForce Ada
specifically — as opposed to its datacenter siblings — runs FP32-accumulate
GEMMs at *half* the throughput of BF16/FP16-accumulate ones. Every one of
these is a documented hardware fact we had to design around rather than a
theoretical concern: they're the reason "just cast everything to FP8" (the
first thing anyone would try on Hopper) was never a live option here, and the
reason so much of this project's engineering time went into working out what
*is* available on this specific chip rather than assuming a datacenter
recipe would transfer.

**A finite AI token budget, not an unlimited one.** The other resource
constraint was ours to manage directly: every profiling pass, every
subagent dispatch, every re-verification costs tokens, and this was run on a
constrained plan, not an enterprise API budget. Raw `ncu`/`nsys` output is
25,000-100,000 tokens per capture — reading that directly, every time, for
every candidate, would have burned the entire budget on profiling alone
before a single optimization shipped. We designed around that from the start
rather than discovering it the expensive way: we sketched a full six-role
agent architecture (orchestrator, profiler, strategist, implementer,
verifier, adversary) and then deliberately did not deploy it, because we
calculated it would cost roughly *11x* more tokens per iteration than the
two-role version we actually ran (a cheap fact-only profiler plus a
verification discipline we enforced ourselves, escalating to an expensive
implementer agent only for the small number of problems that genuinely
needed one). That decision — recorded in our own internal `docs/AGENTS.md`
before writing a line of optimization code — is the reason we could afford
to run ~20 real, independently-verified investigations in the time we had.
Every design choice in the next section follows from taking that budget as
seriously as we took the GPU's.

## Build Environment

| Component | Detail |
|---|---|
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-30-generic |
| CPU | AMD Ryzen 7 7700 (8 cores / 16 threads), 29 GB RAM |
| GPU | NVIDIA GeForce RTX 4090, 24 GB, driver 580.173.02 (consumer/workstation tier — see above) |
| Host CUDA | 13.0 |
| Container toolchain | CUDA 13.1.80 (nvcc), g++ 13.3.0, PyTorch 2.13.0+cu130, cuDNN 9.2.0 |
| Isolation | Apptainer 1.4.5, `--nv --cleanenv` |
| Scheduler | Slurm 25.11.2, single-node, `--exclusive --gres=gpu:1` per job |

Every benchmark and every accuracy check ran inside this container, dispatched
exclusively through `sbatch` — never a direct `python` invocation on the GPU
node. That's not a style preference: Slurm's job prolog is responsible for
clock-locking the GPU, and a direct run bypasses it, silently corrupting every
timing number that follows. We treated that as a hard invariant throughout,
and it mattered in practice — on a single shared node, a job can queue behind
someone else's session, and clock/thermal drift across a long work session is
real and measurable (we caught it doing exactly that; see below).

## The Agentic Workflow

This is the part of the project we think matters most, more than any single
speedup number, because it's what made the speedups trustworthy under the
constraints above.

### The iteration discipline

Every candidate optimization went through the same fixed pipeline, one change
at a time:

```mermaid
flowchart TD
    A[Pick ONE candidate,<br/>cite a specific profiled fact] --> B[Static validity gate<br/>tools/check_validity.py — free, no GPU]
    B -->|fails| Z1[Fix or discard<br/>before spending GPU time]
    B -->|passes| C[Smoke test: 1-2 shapes]
    C --> D[Full accuracy sweep,<br/>all 8 shapes, via sbatch]
    D -->|any real failure| E[Diagnose root cause,<br/>document, revert]
    D -->|passes| F[Full benchmark sweep]
    F --> G{Real, reproducible<br/>speedup vs. shipped state?}
    G -->|no / marginal| E
    G -->|yes| H[Archive as new elite<br/>MAP-Elites: regime × family]
    H --> I[Document with exact<br/>numbers + commit]
    E --> I
    I --> A
```

We never bundled two optimizations into one candidate — a bundled diff that
fails leaves you unable to tell which half broke it, and every re-test costs
real GPU queue time on a shared, single-node cluster and real token budget to
re-diagnose. The static gate exists specifically to catch invariant
violations — caching an output on an input tensor's pointer, registering a
fused weight as a `Parameter` (which breaks `load_state_dict(strict=True)`),
an explicit attention mask silently disabling the fast attention kernel path
— for zero GPU cost and zero token cost, before any job is even submitted.

### Splitting the work: cheap facts, expensive engineering

This is the direct product of the token-budget constraint above. We isolated
all profiling in a lightweight subagent (Haiku-tier) whose only job is:
submit the profiling job, parse the raw `ncu`/`nsys` output into a compact
JSON fact block (top three hot kernels, occupancy, throughput as a percentage
of the hardware's real peak), and return nothing else — no advice, no
proposed fix, and the 25,000-100,000 tokens of raw CSV never reach the main
context at all. That single design choice turned a context-devouring
operation into a net token *saver*, and it's what made it affordable to
profile before nearly every real decision instead of guessing.

The genuinely hard work — writing real CUDA/C++ extensions against cuBLASLt's
raw API, building and measuring a fused-kernel prototype, diagnosing why a
Triton kernel loses to cuBLAS — went to a smaller number of Opus-tier
implementer agents, each given a complete, self-contained brief rather than
an open-ended prompt: the specific hypothesis to test, every relevant prior
finding (so they wouldn't waste a GPU job and a token budget re-discovering
something already measured), the exact technical facts needed (verified CUDA
struct layouts, known toolchain gotchas), and an explicit instruction to
report a clean negative honestly rather than force a marginal result to look
like a win. Spending more tokens per dispatch on the few problems that
genuinely needed deep, expensive reasoning — instead of spreading an
elaborate six-role architecture thin over every problem regardless of
difficulty — is what made the constrained budget stretch across a whole
session of real investigations.

### Verification as the actual differentiator

Every subagent's self-reported result got checked against the raw log file
before we trusted it — not because the agents were unreliable, but because
the whole project's currency is verifiable claims, and a self-report is not
one until it's been checked. Four moments from this session illustrate why
that discipline mattered:

**We caught a false-positive benchmark inside an agent's own result.** An
implementer reported a 1.26–1.96x win from a cuBLASLt algorithm search on the
attention path's FP16 GEMMs. Its own instrumentation — timing the same
operation two independent ways, a CUDA-graph replay and a `torch.profiler`
kernel-time capture — showed both sides were dispatching the *literal same
kernel* (`maxdiff = 0.0`), and the "speedup" was actually the gap between two
measurement harnesses' dispatch floors (~7.8 µs Python-loop overhead vs. ~3.9
µs C++-loop overhead), not a real kernel difference. One kernel cannot be
1.96x faster than itself. Caught before it was ever proposed for shipping.

**We caught cluster drift disguised as a regression.** After shipping a
change scoped to only affect one shape, a full sweep showed a different,
untouched shape had apparently regressed 6%. Rather than accept or dismiss
that, we re-ran the *prior, unmodified* code under current conditions — it
reproduced the identical "regression." The cluster's clock/thermal state had
simply drifted over the session; the code was innocent. That became a
standing rule for the rest of the project: never trust a delta under ~10%
against an old logged number — get a fresh, same-session baseline first.

**We ran a real experiment instead of trusting an inference.** A profiling
pass suggested launch overhead was already at zero after CUDA graphs,
implying a hand-fused "megakernel" wouldn't help. Rather than close the
question on that inference alone, we built the direct test: using cuBLASLt's
in-place vs. out-of-place split-K reduction to physically delete one real
kernel launch while holding the algorithm and tiling fixed. Result: removing
that boundary cost 3–9x more than the 0.86 µs launch it eliminated. That's a
measurement, not an estimate, and it's the strongest evidence in the project
for why the megakernel investigation stopped where it did.

**We caught a near-miss that a smaller sample size would have missed.** A
precision-reduction candidate passed a 5-trial accuracy check cleanly. Because
its worst-case error sat above our own "investigate further" threshold, we
re-ran it at 40 trials — and found a real ~30%-of-trials failure rate hiding
behind the lucky 5-trial pass. Reverted before it ever reached the shipped
model.

### Documenting failure as rigorously as success

Of roughly twenty real investigations this session, more than half closed
without shipping anything. Every one is recorded with the same standard as a
win: what we tried, what we measured, why it didn't survive, and what would
have to change for it to be worth revisiting. We treat that record as an
asset, not a tally of failures — it means nobody on this project, human or
agent, re-tries a dead end and re-spends the token or GPU budget finding out
again, and the reasoning behind every "no" is preserved alongside every
"yes."

## Optimizations

### Shipped

- **FP16 attention path.** QKV projection, attention, and output projection
  run in FP16 (residual stream, softmax, and LayerNorm stay FP32 — never
  quantized). Verified clean at 40-trial accuracy rigor on every shape, with
  zero failures even at the largest tested tensor (671 million elements).
  This is the single biggest lever in the project: at full FP32 precision,
  PyTorch's scaled-dot-product-attention is structurally locked out of its
  flash and memory-efficient kernels and falls back to a generic math
  implementation. Casting the attention path to FP16 isn't just a smaller
  number format — it's the key that unlocks a fundamentally different, much
  faster attention kernel, which is why long-sequence (where attention
  dominates the forward pass) shows the largest gain in the table above.
- **Explicit cuBLASLt algorithm selection for the feed-forward GEMMs, tiny
  batch only.** cuBLASLt's own default heuristic doesn't select the fastest
  available kernel for this specific small-M shape; a one-time eager
  calibration (never inside the compiled/graphed region) picks a split-K
  variant that measurably beats it, with a correct fallback to the standard
  path if the calibration doesn't actually win on the machine it's running
  on. This is most of the extra gap between the tiny row's 7.37x and the
  large-batch row's 1.97x above.
- **A hand-written warp-specialised `mma.sync` GEMM, FP32-accumulate, with a
  fused epilogue.** For large token counts (`≥ 8192`) the attention
  projections (non-causal) and the feed-forward input GEMM (causal) run on a
  custom kernel with a producer/consumer named-barrier pipeline and a
  128-bit shared-memory-staged epilogue, in place of cuBLAS. It accumulates
  in FP32 — same numerical precision as the library path — so it spends no
  accuracy budget, and for the causal FFN it *also* fuses the exact erf-form
  GELU into the epilogue, computed bit-for-bit identically to the reference,
  collapsing a GEMM plus a full-tensor activation pass into one kernel.
  Non-causal: +4.8% / +5.4% at long-sequence / large-batch. Causal: +9.5% /
  +12.1% at long-sequence / large-batch (d512/ffn2048), with the model's
  output error provably unchanged to the last bit. This is the first place
  in the project where a hand-written kernel beats the vendor library on a
  shipped path — the earlier FP16-*accumulate* attempts at the same kernel
  (see "Investigated and closed") were real speedups that the accuracy
  budget made uncollectible; moving the accumulation back to FP32 and paying
  for the win with pipeline engineering instead is what made it shippable.
- Underneath all of these: SDPA in place of manual attention math, fused
  QKV projection, an exact power-of-two scale fold, and CUDA graphs via
  `torch.compile(mode="reduce-overhead")` for launch-overhead elimination —
  the foundation the FP16/cuBLASLt/warp-spec work above builds on.

### Investigated and closed

Several of these were built as real, working artifacts before being rejected
on measurement — not skipped on suspicion, and each one exists specifically
because the consumer-GPU constraint above ruled out the more obvious
datacenter-style shortcut.

| Candidate | Looked promising because | Closed because | Evidence |
|---|---|---|---|
| BF16 / FP8 / INT8 everywhere | Standard precision-reduction playbook | All three miss the accuracy budget outright — BF16 by ~11x, INT8 by ~27-31x, FP8 (even with correct per-channel scaling) by ~65-78x; each format fails more badly than the last, not better | 20-seed accuracy probes, real hardware `torch._scaled_mm`/quantize-dequant tests |
| `torch.compile(mode="max-autotune")` | Free kernel search, keeps CUDA graphs | Mixes in Triton GEMM kernels that emulate TF32 differently from cuBLAS's native path — enough rounding drift to fail accuracy on the very first shape tested | Full 8-shape sweep, decisive failure |
| L2 cache persistence via a real CUDA/C++ extension | Model's hot working set (~63 MB) is close to L2 capacity (72 MB) | Ada caps the *persisting* L2 partition at 49.5 MB; the real working set already sits comfortably in normal LRU without help — the persistence window itself cost 4-6% for zero measured benefit | A working pybind11 extension against real CUDA 13.1 headers, isolated probe + full-sweep decomposition |
| Fused FFN kernel (Triton) | Removes ~48 MB of intermediate-activation DRAM traffic per forward pass | The FFN is compute-bound (60-69% of the TF32 roofline already), not memory-bound — that traffic was already fully hidden behind compute; fusion only sacrifices cuBLAS's per-GEMM tiling freedom | Kernel built, verified against float64 ground truth, measured 0.18-0.87x (never a win) across every shape tested |
| Full megakernel / kernel-fusion investigation (the Hopper-style persistent-kernel idea, adapted for Ada's `cp.async` instead of TMA) | Fewer kernel launches should mean less overhead | CUDA graphs had already reduced launch overhead to effectively zero; a direct experiment (deleting one real kernel boundary via cuBLASLt's in-place split-K) showed removing it costs 3-9x more than it saves | Fresh kernel census, a decisive built experiment, four independent readings of the "launch overhead" gate |
| GELU polynomial approximation | Catalogued as a low-risk, low-cost win | The catalogued accuracy estimate for a degree-7 polynomial was simply wrong by ~5 orders of magnitude once actually computed — verified by direct numerical fitting, no GPU needed | Chebyshev/minimax fit computed and checked against the exact function |
| Hand-written `mma.sync` PTX, FP16 accumulate | cuBLAS is capped at FP32 accumulation on this GPU; a different accumulation tier is a mechanism cuBLAS can't offer at all, not just a different algorithm within its existing tier | The mechanism is real (an isolated accumulate-type-only A/B on the identical kernel measured 1.4-1.5x) but the raw GEMM win was only 1.2x, diluting to a ~2% whole-model gain at best — and a later, more thorough pass (see the CUTLASS row below) established that even that gain was never collectible: the accumulation tier fails this model's accuracy budget outright, at every shape that would have benefited. **The kernel itself was not wasted** — its FP32-accumulate configuration plus warp-specialisation later shipped (G4.3 non-causal, G4.7 causal; see "Shipped"), trading the unaffordable accumulation tier for pipeline engineering | First raw PTX in the codebase; fragment-level correctness verified three independent ways before any performance work; a 26-configuration tile search across three independent rounds, cross-checked via CUDA-graph replay and profiler kernel time in agreement to within 1-3% |
| SASS-level hand-tuning via `CuAssembler`, informed by the `CuAsmRL` research paper's instruction-reordering approach | The remaining efficiency gap in our hand-written kernel needed exactly the kind of fine-grained scheduling control this tooling promises, without a full kernel rewrite | Not an Ada problem — the tool correctly recognizes and encodes our GPU's instruction set (verified: it re-encoded every instruction our kernel uses and matched the original bytes exactly, zero errors). The actual wall is that our CUDA toolkit's binary container format has moved on from what the tool's file-format reader expects, at one specific internal section it can't reconstruct from text. We fixed two smaller incompatibilities before hitting this one, and confirmed it isn't specific to our GPU by reproducing the identical failure on an older, officially-supported architecture | A real dependency install and toolchain probe, not a documentation lookup; the specific failure point isolated precisely enough to know exactly what would need to be built to route around it |
| NVIDIA's own production kernel-template library, hand-configured for the exact GEMM shapes and accumulation tier our earlier kernel proved was real, pushed all the way through to a full accuracy verdict | Sidesteps the SASS-editing wall entirely (compiled fresh from source, never parses a pre-built binary's container format) and directly targets the structural efficiency gap our hand-written kernel was missing | **Closed on accuracy, not speed — the more decisive of the two possible answers.** The kernel itself is provably correct (its exact-FP32-accumulate configuration reproduces our baseline's numbers to all seven printed digits — two independent implementations agreeing bit-for-bit); the accumulation tier itself is what's unaffordable. It only had 9% of error budget left after ordinary FP16 storage had already spent the rest, and this tier costs roughly 7-8x more error than that at the reduction depth these matmuls need — real headroom, but nowhere near enough. We checked whether isolating just one of the two matmuls could route around it: the closest variant passes four of six affected shapes, but fails on the exact one shape that had any speed benefit to offer, by six elements out of 84 million — so close that it briefly looked like a targeted fix might exist, and precise enough to be sure that it doesn't. This result retroactively answers the open question the hand-written kernel above had left unresolved: the accumulation tier isn't just economically marginal, it's arithmetically unaffordable at this model's precision budget, independent of which tool builds the kernel | A working kernel confirmed correct in isolation, then pushed through the full accuracy suite anyway rather than stopped at a speed verdict; three independent routing configurations tested before concluding no accurate-and-faster subset exists |

## Theoretical Ceiling & Pareto Frontier

**Why the raw FP8 floor (0.122 ms, Default shape) is not the real ceiling.**
CLAUDE.md's own ground truth gives the naive number: 40.27 GFLOP/forward at
the FP8 tensor-core rate of 330.3 TFLOPS (`CUBLAS_COMPUTE_32F` accumulate) —
`40.27e9 / 330.3e12 = 0.122 ms`. That arithmetic is correct but assumes
100% of the model's FLOPs run at that rate. The project's own precision
policy forbids this: "**Never FP8 in attention** — softmax tails die," and
CLAUDE.md quantifies why — FP8's `eps/sqrt(K)` error-averaging argument
needs a large reduction depth to work (the FFN's `K=2048` gives
`eps/sqrt(K) = 6%/45.3 ≈ 0.14%`, comfortably under budget); attention's
per-head reduction depth is `head_dim=64`, giving `6%/sqrt(64) = 6%/8 =
0.75%` — five times worse, and that's before the exponential in softmax
amplifies it further. The policy's own FLOP split says FFN+out_proj (the
part legally eligible for FP8) is 65% of total FLOPs; the remaining 35%
(QKV projection + attention score/context matmuls) is floored at BF16's
165.2 TFLOPS at best:

```
t = (0.65 x 40.27 GFLOP) / 330.3 TFLOPS + (0.35 x 40.27 GFLOP) / 165.2 TFLOPS
  = 26.18 GFLOP / 330300 GFLOP/s        + 14.09 GFLOP / 165200 GFLOP/s
  = 79.3 us                             + 85.3 us
  = 164.6 us  ≈  0.165 ms
```

That is the real, correctness-respecting theoretical floor for Default —
**35% higher than the raw 0.122 ms number**, because 35% of the model's
FLOPs are contractually barred from ever reaching the FP8 rate. (This
derivation deliberately uses the project's original, tighter accuracy bound
— `max_abs=1e-3`/`max_rel=1e-2` — as the stronger, more conservative claim;
the enforced default is `0.002/0.02`, confirmed by the judges' canonical
`torch_transformer_benchmark.py` published 2026-08-27, but the
attention-precision constraint above is a structural policy, not a
tolerance-dependent one — it doesn't move with the budget.) Our shipped
Default result (0.558 ms) sits above even this corrected floor: we use
FP16 storage (not FP8) for both FFN and attention, so there's headroom
left on the table in principle — the FP8 rate is real, but nothing in this
project reached it safely (see below).

**Pareto chart — Speed vs. Numerical Error (Default shape, log-error axis):**

```
speed (ms, log scale, lower=better) →
0.1   0.2   0.3   0.5   0.7   1.0   1.4
 |     |     |     |     |     |     |
 .     .     .     .     .     .     X  BaselineTransformer (1.405ms, error=0 by definition)
 .     .     .     .     X     .     .  UserOptimizedTransformer (0.558ms, max_abs=0.00113) <- SHIPPED, pareto-optimal
 X     .     .     .     .     .     .  mma.sync/CUTLASS FP16-accum tier (~0.09ms modeled, max_abs=0.00763)
 |     |     |     |     |     |     |
-------------------------- accuracy budget ceiling (max_abs=0.002) --------------------------
 ^ FAILS by 3.8x -- fastest point on the chart, but off the feasible region entirely
```

The 660 TFLOPS `mma.sync` FP16-accumulate tier is the fastest theoretical
point in this entire design space (it's what "0.09ms" above stands in for —
proportionally faster than the FP8 rate) — and it is not on the Pareto
frontier, because it isn't in the feasible region at all. We didn't
theorize this: we built it (raw PTX, then CUTLASS's production
implementation of the same tier) and measured a real accuracy failure —
`max_abs=0.00763`, **3.8x over the 0.002 budget**, on the causal-path
attempt (`docs/CAUSAL_LEDGER.md`'s `G4.6c` row); the non-causal attempt
failed by a comparable margin at the old, tighter budget
(`docs/PROGRESS.md` steps 37/40). `UserOptimizedTransformer` is the actual
Pareto-optimal point: every faster point we found and verified either
violated the accuracy constraint outright (this tier) or wasn't actually
faster once measured end-to-end (the fused-FFN-kernel and megakernel rows
above). The one later exception is the **FP32-accumulate** warp-specialised
kernel (G4.3 / G4.7): a different point entirely — it stays inside the
feasible region because it keeps the library's accumulation precision, and
pays for its win with pipeline engineering rather than a cheaper number
format. It moved the shipped point without moving the accuracy.

## Regime Latency Breakdowns

**Methodology, stated plainly:** each stage (FP16 cast+QKV projection,
SDPA, FFN in+out) was timed independently with real CUDA events
(`probes/stage_breakdown.py`, 30 iters post-warmup, per-layer cost x6
layers), called eager (outside `torch.compile`) so each stage's own kernel
cost is isolated. This means the three stages' sum does **not** equal the
top-level Before/After number for that shape — the gap is exactly the
launch-overhead reduction CUDA graphs (G2.4/G2.4b) provide, which only
exists in the compiled/graphed path, not in isolated eager calls. Both
numbers are real and both are reported below, so nothing is glossed over.

| Regime | cast+QKV | SDPA | FFN | eager sum | graphed total (Before/After) | overhead removed by CUDA graphs |
|---|---|---|---|---|---|---|
| Tiny | 0.138ms (27%) | 0.058ms (12%) | 0.309ms (61%) | 0.505ms | **0.201ms** | 0.304ms (60%) |
| Default | 0.140ms (17%) | 0.058ms (7%) | 0.631ms (76%) | 0.829ms | **0.558ms** | 0.271ms (33%) |
| Long-sequence | 0.671ms (11%) | 0.723ms (12%) | 4.865ms (78%) | 6.258ms | **4.600ms** | 1.658ms (26%) |
| Large-batch | 3.464ms (14%) | 0.936ms (4%) | 19.523ms (82%) | 23.922ms | **17.083ms** | 6.839ms (29%) |
| Padded | 0.161ms (18%) | 0.129ms (14%) | 0.628ms (68%) | 0.918ms | **0.564ms** | 0.354ms (39%) |
| Causal | 0.141ms (17%) | 0.057ms (7%) | 0.633ms (76%) | 0.830ms | **0.561ms** | 0.269ms (32%) |

**Reading this honestly, including where it surprised us.** We expected
long-sequence to be SDPA-dominated (O(S²) attention is the textbook story
at S=1024) — the measurement says otherwise: FFN is 78% of eager cost
there too, because the FFN's two GEMMs scale with `B·S` exactly like
attention's compute does at this `d_model`/`head_dim`, and flash/efficient
attention (unlocked by G6.4b's FP16 Q/K/V) is efficient enough that it
never becomes the bottleneck at any tested shape. **FFN dominates every
single regime** (61-82% of eager cost) — which is exactly why `G6.4a_v2`/
`G6.4a_v2c` (FP16 FFN) was the largest single lever in both the non-causal
and causal halves of this project, and why `G6.6` (cuBLASLt) only ever
mattered at Tiny: it's the one regime small enough that launch overhead
(60% of eager cost, largest of any regime) rivals the FFN's own cost, so a
cheaper FFN algorithm shows up in the total instead of being hidden behind
graph replay.

Causal's per-stage split (17%/7%/76%) is now nearly identical to Default's
(17%/7%/76%) — direct evidence the causal-path rewrite (`G0.1c`/`G1.1c`/
`G6.4a_v2c`/`G0.2c`/`G6.4bc`) converged the two paths' efficiency, closing
what used to be the largest gap between any two regimes in this project
(causal started this pass at 1.78x vs Default's 2.13x; both now sit at
2.5-2.8x with near-identical internal structure).

## What we learned

The result we're proudest of isn't the 94% long-sequence number — it's that
every number in this document survives someone else re-running the
underlying job. Working under a real GPU constraint (no Hopper shortcuts
available) and a real token constraint (no unlimited profiling or agent
dispatch) forced a discipline we think produced a better project than an
unconstrained one would have: cheap facts before expensive reasoning, one
verified diff at a time, and never accepting a claim — ours or the model's —
until it had been checked against a second, independent measurement. Twice
this session that discipline caught a result that looked like a genuine win
and wasn't. We'd rather ship a verified 2.52x than an unverified 3x.

## What's next

We went ahead and tested the highest-risk hypothesis we had — hand-written
`mma.sync` PTX reaching for the FP16-accumulate tier cuBLAS cannot touch on
this GPU, a mechanism no algorithm search within cuBLAS's own tier could ever
find, since cuBLAS is architecturally capped at FP32 accumulation here. This
was the first raw-PTX kernel written in the project, built and validated in
stages (fragment-level correctness verified three independent ways before any
performance work) rather than committed to blind. The mechanism is real —
an isolated accumulate-type-only comparison on the otherwise-identical kernel
measured a genuine 1.4-1.5x — but it didn't convert into a shippable win:
cuBLASLt already extracts 91% of its own tier's ceiling, our first
hand-written kernel reached only 55% of its higher one, and after diluting
that gap through the one shape where it applies at all, the honest number is
roughly a 2% whole-model gain — decisively below the bar we set for every
other change in this document. Getting further would mean CUTLASS-grade
kernel engineering (shared-memory-staged epilogues, warp specialization), not
another tuning pass.

We tried to route around that specific wall at the machine-code level next,
using `CuAssembler` — a community tool for disassembling and re-editing
compiled GPU binaries directly — informed by a recent research paper's
approach to automatically finding better instruction schedules this way. That
investigation answered a real, previously-untested question along the way:
our GPU's instruction set is not the problem (confirmed by getting the tool
to correctly re-encode every instruction our kernel uses, byte-for-byte), but
our specific CUDA toolchain version has moved its binary file format past
what the tool's file reader expects, at one section we couldn't reconstruct
without building a much larger piece of infrastructure than the win at stake
justified.

Rather than stop there, we pursued the more direct route to the same
"CUTLASS-grade kernel engineering" gap: hand-instantiating NVIDIA's own
production kernel-template library for the exact GEMM shapes and the specific
accumulation tier our earlier kernel proved was real but under-realized.
Notably, the kernel our baseline already uses for these shapes turns out to
be built from an older generation of that same library — this wasn't a shot
in the dark, it was a targeted attempt to out-configure what's already
shipped with a newer configuration of the tool that built it.

It worked, partially, and precisely. The library compiled cleanly against
our exact setup with none of the file-format friction that stopped the
previous attempt (exactly the outcome we'd reasoned our way to: a tool that
compiles fresh from source has nothing to break against an evolved binary
format). Configured for our shapes and searched across two rounds, it beat
our own hand-written kernel by 1.3x and our baseline by up to 1.6x — a real,
independently-confirmed number, not a lucky reading. It settled at roughly
72% of the accumulation tier's theoretical ceiling, short of an 80% bar we'd
set for ourselves in advance, for a reason we could show precisely: closing
that stretch needs a pipelining technique the library only implements for
newer, non-consumer GPU generations. We confirmed the search itself had
converged rather than stopped early before accepting that number.

We didn't stop at that speed verdict. Rather than treat a near-miss on a
self-imposed bar as the end of the story, we asked the harder question the
speed number alone couldn't answer: even granting the smaller, real gain on
the table, would it actually be usable? We verified the kernel was
computing the right answer — its exact-precision configuration reproduced
our baseline's output to all seven printed digits of accuracy, two
independent implementations agreeing bit for bit — and then ran it through
our full accuracy suite anyway. That's where the real answer was. The
faster accumulation tier costs roughly seven to eight times more numerical
error than our baseline at the depth these particular matrix multiplications
require, and after ordinary lower-precision storage had already spent most
of our error budget, there wasn't seven-to-eightfold headroom left — there
was about a tenth of that. We checked whether isolating just one of the two
operations involved could route around the shortfall; the closest variant
got within six elements out of eighty-four million of passing on the one
shape that had any speed benefit to offer, and failed on precisely that
shape. Close enough to be worth checking, precise enough to be sure no
accurate-and-faster version exists.

That finding closes the loop on the earlier hand-written attempt too: it
had left this exact question open, and now it's answered for both
implementations at once. The faster accumulation tier is not a
kernel-quality problem or a tuning problem — it is arithmetically
unaffordable at this model's precision budget, full stop, independent of
which tool builds the kernel.

Every angle we identified — compiler-level flags, alternative precisions,
cache residency, kernel fusion at every granularity we could measure,
hand-written machine code, machine-code editing tools, and a production
template library pushed all the way through to a final accuracy verdict —
has real hardware evidence behind it, in either direction. What's left from
here is not a technique we haven't tried; it's hardware this project's GPU
doesn't have.
