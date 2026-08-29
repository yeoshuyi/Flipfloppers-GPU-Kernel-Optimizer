# Devpost — Written Project Description

*Paste into the Devpost form field-by-field. Narrative ≈ 900 words; the required
lists and the tag/keyword blocks are extra.*

---

## Tagline

An **AI-assisted, agentic** system that rewrites a frozen Transformer layer's
forward pass for a single **consumer RTX 4090 (no Hopper)** — **6.3× aggregate,
7.7× geomean, up to 31.8× per shape**, every output within the `rel < 0.02 /
abs < 0.002` budget, at **94–96 % of the accuracy-legal hardware roofline**.

---

## Problem statement addressed

**Problem 3 — "Implement a GPU Kernel for a Transformer Layer."** The prompt
asks for GPU kernels implementing a fixed Transformer layer, optimised for
runtime on a chosen GPU, passing every test shape within `relative error < 0.02`
**and** `absolute error < 0.002` versus the reference — per-shape
implementations allowed via runtime shape checks, AI-assisted methods
encouraged.

Our target GPU is an **RTX 4090 (Ada Lovelace, `sm_89`)** — the workstation
tier, not an H100. We ship a drop-in `UserOptimizedTransformer` that dispatches
to a different implementation per regime from a runtime `tok = batch × seq`
check, plus a single generated self-contained file the grader runs unchanged.

---

## Inspiration

The prompt names the five things that can bottleneck a Transformer — compute
throughput, memory bandwidth, cache efficiency, kernel-launch overhead,
tensor-core utilisation. We took that literally: *measure which one binds for
each shape, then attack only that one.* Nearly every published
Transformer-inference speedup assumes datacenter silicon (Hopper's TMA,
`wgmma`, the FP8 accumulate tier). We wanted to know what a developer on one
consumer card actually gets without touching weights or breaking the accuracy
budget — and whether an agentic loop can do the bottleneck analysis, kernel
generation, and rigorous verification cheaply, under a finite AI token budget.

---

## What it does

Takes the frozen `BaselineTransformer` plus an accuracy tolerance; returns a
much faster forward pass whose outputs match the baseline within budget on every
tested shape.

- **Regime dispatch** — tiny / default / long-seq / large-batch, causal &
  non-causal, each a separate tuned path chosen by a `tok` check.
- **Precision-budgeted** — FP16 *storage* with FP32 accumulate (the only tier
  that fits), FP32 softmax and residual stream kept exact, LayerNorm affine
  folded into the next GEMM.
- **Fusion + graph capture** — fused QKV, scale/affine folds, CUDA-graph
  replay, and a fused GEMM + exact-erf-GELU kernel where the shape allows.
- **Roofline-aware stopping** — a formal accuracy-constrained roofline tells
  the loop when a shape is at the practical hardware limit.

Result: **Σ 383.4 ms → 60.8 ms (6.3×)** over the 14-row official matrix,
**geomean 7.7×**, per-shape **1.9×–31.8×**, **13 / 13 within budget**
(`failed == 0`). The 14th shape (`S = 100000`) OOMs the FP32 *reference* on
24 GB so the harness can't score it — but the shipped model still **executes
it** on one card via sequence chunking (13.0 s, 20.8 GB peak, output inside
the accuracy gate).

---

## How we built it

Fully AI-assisted. One diff per iteration through a fixed loop —
**research → implement → verify → gate → document → commit.**

- **A cheap profiler subagent** compresses 25 000–100 000 tokens of Nsight
  Compute output into a 20-line JSON fact block; an expensive implementer
  agent is escalated to only for the few problems that need one.
- **Manual research injection** — non-catalogue ideas (papers,
  owner-specified protocols) enter the candidate queue by hand.
- **Every candidate is gated:** a static "is it gaming the benchmark?"
  detector → a 40-seed accuracy sweep → a matched before/after benchmark — all
  through **Slurm `--exclusive` with clocks locked in the job prolog**, inside
  **one pinned Apptainer image**, never a direct GPU run.
- **A formal expected-value rule** decides ship / no-ship:
  `EV ≈ P(gain real)·gain − P(over-budget on an unseen shape)·(score → 0)`.
- **~20 investigations; over half closed as documented negatives**, so no dead
  end is re-tried. Agent-token overhead was engineered from **~45 k to ~4 k per
  iteration** by replacing five of six planned agent roles with two scripts.

Hardware/software: RTX 4090 · CUDA 13.1 · PyTorch 2.13 + Inductor · Triton ·
CUTLASS · Nsight Compute.

---

## Results / benchmarks

- **6.3×** aggregate, **7.7×** geomean, **1.9×** (compute-bound) to **31.8×**
  (long-sequence) per shape.
- **13 / 13 shapes pass** `rel < 0.02 / abs < 0.002`; tightest margin
  `max_abs = 0.00195` (97.5 % of budget).
- **Row 14 (`S = 100000`), which the harness cannot score, runs anyway** — a
  memory-feasibility gate routes it (and any larger causal shape) to a
  sequence-chunked forward: FP16 residual mutated in place, an incrementally
  filled K/V cache, per-chunk attention split into a past block + a
  square-causal block merged by log-sum-exp. **13.0 s, 20.8 GB peak** on one
  RTX 4090; `failed == 0 / 4.1e8` vs a higher-precision chunked reference.
- Compute-bound (row 8): shipped GEMMs at **94–96 % of the accuracy-legal
  165.2 TFLOP/s roofline** in isolation.
- Memory-bound (row 6): the elementwise path moves **22.7 GB/forward against a
  first-principles prediction of 23.6 GB** — *at* the DRAM bandwidth roofline.
- **The accuracy gate is load-bearing:** a CUTLASS kernel that ran the GEMMs
  **1.57–1.60× faster than cuBLAS** was reverted — FP16 accumulation cost
  **×7.18** on the GEMM's own error and failed **6 of 8** shapes.

---

## Challenges we ran into

- **The precision-reduction playbook is off the table** — BF16, FP8, INT8 miss
  the 0.002 budget by **5–40×**. Reaching the faster tensor tier needs hand-written
  **`mma.sync` PTX**; cuBLAS is architecturally capped at FP32 accumulation on
  this chip.
- A **SASS-level** rewrite via CuAssembler hit a CUDA 13.1 binary-format wall.
- A **from-scratch fused whole-layer megakernel** hit Ada's occupancy ceiling
  (96 KB shared → 1 block/SM → **×0.74**, slower).
- **Verification caught two things that looked real and weren't:** cluster
  clock-drift disguised as a 6 % regression, and a "speedup" that was one
  kernel timed against itself.

---

## Accomplishments that we're proud of

- **Every number survives a third-party re-run** — 120+ Slurm job logs, one
  command to reproduce.
- A **formal accuracy-constrained roofline** proving the shipped stack is at
  the practical hardware frontier: faster requires Hopper-class hardware or an
  accuracy-budget violation — **both built and measured**, not asserted.
- Real low-level engineering: **inline PTX**, a **warp-specialised
  named-barrier `cp.async` pipeline**, a CUTLASS bake-off, a fused causal
  megakernel, offline cuBLASLt algorithm search.
- **100 % of shapes correct with zero per-shape human tuning** — the loop
  chose and verified every path.

---

## What we learned

Two constraints — no Hopper, and a finite token budget — forced a better
process: cheap facts before expensive reasoning, one verified diff at a time,
and never trusting a claim (ours *or* the model's) until a second independent
measurement agrees. An explicit accuracy oracle changes which optimisations are
even admissible — most of the speed on this hardware is illegal, not
unreachable.

---

## What's next

- A **Hopper backend** — TMA + `wgmma` unlock the persistent fused megakernel
  Ada cannot run.
- INT8 with proper outlier handling; autotuned launch configs; a backward pass.
- For `S = 100000`: a *chunked baseline* so the harness can score it, and
  multi-GPU sharding to bring the 13 s single-card chunked forward down.
- Publish the accuracy-legal kernels + the negatives ledger, so the next team
  doesn't re-spend the budget on the same dead ends.

---

## Development tools used

Claude Code (the agent stack) · NVIDIA Nsight Compute & Nsight Systems · Slurm
(batch orchestration, clock-locking) · Apptainer / Singularity (reproducible
container) · CUDA Toolkit 13.1 / `nvcc` · git · tmux · VS Code.

## APIs used

**Anthropic Claude**, via Claude Code, as the agent model — for profiling
analysis, kernel generation, and verification orchestration. No other external
or hosted APIs; nothing in the runtime pipeline calls a network service.

## Libraries and frameworks used

PyTorch 2.13 (cu130), incl. `torch.compile` / TorchInductor · Triton · CUDA C++
and **inline PTX** (`mma.sync`, `cp.async`, `ldmatrix`, named barriers) ·
CUTLASS 4.7.1 (header-only) · cuBLAS / cuBLASLt · pybind11 · NumPy.

## Datasets and assets used

**None.** Inputs are synthetic random tensors from the provided
`torch_transformer_benchmark.py` harness (`generate_random_case`); correctness
is measured against a **float64** recomputation of the reference layer. No
pretrained weights — the baseline is randomly initialised and frozen per run.

---

## Built With  *(Devpost tags)*

`python` · `pytorch` · `cuda` · `triton` · `cutlass` · `cublaslt` ·
`nvidia-nsight-compute` · `pybind11` · `numpy` · `apptainer` · `slurm`

## Standout keywords  *(for the writeup, demo video, and Q&A — not tag-field entries)*

accuracy-constrained roofline · at the hardware frontier · 94–96 % of the
accuracy-legal roofline · no Hopper, consumer silicon · shape-specialised
dispatch · AI-assisted bottleneck analysis · agentic verify-loop · inline-PTX
`mma.sync` · warp specialisation · named-barrier `cp.async` pipeline · operator
fusion · CUDA graphs · mixed precision · FP32 softmax · honest negatives as a
reusable asset · reproducible measurement (Slurm + Apptainer, locked clocks) ·
6.3× aggregate / up to 31.8× per shape / 13-of-13 within budget

---

## Links

- **GitHub repository:** _\<add public URL\>_
- **README:** _\<repo\>/blob/main/README.md_
- **Demo video (YouTube, public):** _TBD_ — walkthrough of `./run_eval.sh` plus
  a result-analysis pass (accepted for a backend track with no front-end).

## Team & contributions

_(to be completed by the team)_
