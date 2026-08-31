# GPU Kernel Acceleration of Transformer Algorithm on RTX 4090
> Submitted for Tiktok TechJam 2026.

> Kernel optimization of a Transformer Algorithm on consumer grade single-node RTX 4090 through an Agentic AI auto-research loop. Achieved performance improvements of **1.93x - 31.76x** over various test shapes, with **geometric-mean speedup ≈ 7.7×**

![RTX 4090 sm_89](https://img.shields.io/badge/GPU-RTX%204090%20·%20sm__89-2b6cb0)
![CUDA 13.1](https://img.shields.io/badge/CUDA-13.1-76b900)
![PyTorch cu13x](https://img.shields.io/badge/PyTorch-2.13%20cu130-ee4c2c)
![Apptainer](https://img.shields.io/badge/runtime-Apptainer-1a5276)
![reproduce](https://img.shields.io/badge/reproduce-.%2Frun__eval.sh-3c7a56)

---
## Key Features

1. **We hit the hardware limit on the RTX 4090**. On the compute-bound shape, the optimized GEMMs runs at **94-96%** of the 165.2 TFLOP/s legal roofline. On the memory-bound shape, the elementwise path moves **22.7GB/forward against the structural limit of 23.6GB/forward**, which is the DRAM bandwidth roofline. The remaining gap can be attributed to non-removable runtime overheads.

2. **Built by Agentic Loop**. An agentic system was made to research, implement, verify and gate one diff at a time, with complex heuristics to balance performance and accuracy tradeoff. **Token budget mechanisms** were implemented through model selection and tiered implementation with low cost smoke checks. The agentic system was proficient in optimizing at the **PTX and SASS level with low level, complex toolchains like cuBLAS, CUDA, CUTLASS and CuAssembler.**

3. **Comprehensive Validation Loop**. Each run of validation runs on a pinned `Apptainer` container with `Slurm` used to reserve the **same fixed resource and persist GPU clock** and audit logging. Fair benchmarking achieved through multiple runs with random seeds, measured against GPU-side elapsed time. **Memory bias** is countered by alternating execution order between models to prevent L2 cache bias. **Anti-Gaming** through `check_validity.py` script to prevent models from caching answers from previous runs.

4. **Low Level Engineering**. Utilised AI-written:
> * Inline-PTX `mma.sync` GEMM
> * Warp-specialised named-barrier producer/consumer pipeline with `cp.async`
> * `CUTLASS` implementation comparison
> * `Fused Causal MegaKernel` from scratch
> * `cuBLASLt` algorithm search


5. **Shape Based Strategy Arbiter**. Shapes are dynamically dispatched to different optimization strategies based on the per-shape benchmarking done at each validation round. This ensures that the best optimization techniques are used for each shape. The Arbiter conditions are **fine tuned accurately to the break-even point** (where we see performance gain). See [Shape Arbitration](#shape-arbitration) for the full dispatch, including the memory gate that routes `S = 100000` onto the chunked path.


## Results
Evaluated by running the generated judge drop-in `torch_transformer_benchmark.py` — the same file a grader executes — on the fully-causal test shapes at `atol = 0.002`, `rtol = 0.02`. All 13 scorable test shapes pass the accuracy gate. Row 14 (`S = 100000`) cannot be scored by the harness at all — its FP32 reference OOMs a 24 GB card before our model is ever called — but the shipped model still executes it.

>**Row 14 is un-runnable on the provided benchmark**. The input tensor physically OOMs the GPU on the provided benchmark script. Our optimized model solves this by **sequence chunking** against an incrementally-filled FP16 cache. However, we are not able to test performance against the benchmark. Thus, we compare performance against a sequence chunked version of the benchmark.

> The figure shows **speed increase** compared to the benchmark. The latency is further broken down into the 4 stages: SDPA, GEMM, GELU and LayerNorm.

![per-shape speedup](assets/results.png)

<details>
<summary>Full results table of the 14 Test Shapes</summary>

| # | B | d | H | S | baseline | **shipped** | speedup | max_abs |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 64 | 128 | 4 | 128 | 1.0445 ms | **0.2120 ms** | 4.93× | 0.00137 |
| 2 | 1 | 128 | 4 | 128 | 1.0846 ms | **0.0778 ms** | 13.94× | 0.00137 |
| 3 | 4 | 128 | 4 | 128 | 1.0496 ms | **0.0881 ms** | 11.92× | 0.00137 |
| 4 | 16 | 128 | 4 | 128 | 1.0434 ms | **0.1116 ms** | 9.35× | 0.00137 |
| 5 | 128 | 128 | 4 | 128 | 1.7050 ms | **0.3727 ms** | 4.57× | 0.00137 |
| 6 | 10000 | 128 | 4 | 128 | 290.53 ms | **52.484 ms** | 5.54× | 0.00195 |
| 7 | 64 | 32 | 4 | 128 | 1.0391 ms | **0.1034 ms** | 10.05× | 0.00211 |
| 8 | 64 | 1024 | 4 | 128 | 8.3681 ms | **4.3333 ms** | 1.93× | 0.00141 |
| 9 | 64 | 128 | 1 | 128 | 0.9572 ms | **0.2099 ms** | 4.56× | 0.00145 |
| 10 | 64 | 128 | 2 | 128 | 1.0453 ms | **0.2130 ms** | 4.91× | 0.00138 |
| 11 | 64 | 128 | 16 | 128 | 4.3858 ms | **0.2857 ms** | 15.35× | 0.00137 |
| 12 | 64 | 128 | 4 | 32 | 1.0363 ms | **0.1229 ms** | 8.43× | 0.00141 |
| 13 | 64 | 128 | 4 | 1024 | 70.153 ms | **2.2108 ms** | 31.73× | 0.00137 |
| **14** | 32 | 1024 | 16 | 100000 | **OOM** | **9952.6 ms** | n/a | n/a |

All 13 scorable rows from one run of the generated judge drop-in
(`results/logs/official_causal_sweep_run216.log`, `python3
torch_transformer_benchmark.py --causal ...`). `max_abs` is **byte-identical to
`run168`** on every row — the entry-point switch changed no arithmetic — and no
row moved more than `+0.52%` in latency.

† Row 14 is **not** a harness result. Its baseline cell is `OOM` because the
FP32 reference cannot be constructed at all. The harness dies in
`generate_random_case` at `x * input_scale`, before either model runs
(`results/logs/row14_extreme_run172.log`) — so there is no speedup to quote.
`9952.6 ms` / `19.55 GB peak` are from `experiments/g7_0_chunked_oversize.py`
(job 211). [How Row 14 is benchmarked and
accuracy-checked](#how-row-14-is-benchmarked-and-accuracy-checked).

</details>

### How Row 14 is benchmarked and accuracy-checked

Row 14 is unscorable, so every number for it comes from our own driver
(`experiments/g7_0_chunked_oversize.py`) rather than the harness. That driver is
the harness **extended**. The
claim that its numbers are as trustworthy as a scored row rests on three
measured legs, not on assertion.

* 1 — The chunking is algebraically exact, not an approximation.
* 2 — Where the real baseline fits, the chunked path matches it.
* 3 — The only precision spend is FP16 storage, and it is priced at the full
shape.

---

## Contents

- [Context of the Problem](#context-of-the-problem)
- [Optimizations, Accepted and Rejected](#optimizations-accepted-and-rejected)
- [Agentic Auto-Research](#agentic-auto-research)
- [Performance and Accuracy Tradeoff](#performance---accuracy-tradeoff)
- [Setup and Reproducability](#reproducability)
- [Limitations and Future Works](#limitations-and-future-works)

---

## Context of the Problem

`BaselineTransformer` reference applies a pre-norm transformer with a manual `matmul > mask > softmax > matmul` attention. `UserOptimizedTransformer` produces the same forward outputs, within `atol = 0.002` **or** `rtol = 0.02` per element, as fast as possible.

### Our Context
**Development Environment**. Our target is a *single-node RTX 4090 consummer GPU* that runs on the *Ada Lovelace* architecture, with limited compute resource. This constrains us because:
- **No TMA**. Asynchronous global to shared movement must be hand-rolled with `cp.async` and a software pipeline.
- **No `wgmma`**. The load to MMA pipeline must be built from custom synchronous `mma.sync.m16n8k16` PTX.
- **No thread-block clusters or distributed shared memory**. Blocks may only cooperate through global memory.
- **GeForce Ada runs FP32-accumulate GEMMS at half rate**. cuBLAS is architectually capped at FP32 accumulation on this chip, which runs at half rate due to vendor nerfing (sad face).
- **24Gb RAM**. Extremely long sequenced shapes have to be chunked, leading to significant overhead.
- **Full Sequence Processing**. Not Autoregressive Generation, so KV cache optimizations like MLA is not applicable here.

> Thus, the problem shifts from cutting-edge optimization on server-grade architectures, to a **resource constrained hack (truly a hackathon!)**. This truly demonstrates the ability of Agentic AI, being able to optimize specific GPU hardware quirks through **auto research and validation** on actual hardware.

---

## Optimizations, Accepted and Rejected

### The Full Path a Batch Takes

>Every branch is chosen from the shape alone no input values are ever inspected. Hand-written kernels are outlined in
> **amber**, each labelled with the shapes it gets selected for.

![optimization](assets/optimizations.png)

After the layer loop: `final_norm` (not folded — it has no consumer). The
FP32 residual stream is never downcast, both LayerNorms do only the mean/var
reduction, `ffn_out` deliberately stays at ≈FP32 (TF32x3).

### Accepted Optimizations
All applied optimizations are exact or precision-neutral, with the exception of two FP16-storage casts. These casts are strictly gated by an `atol = 0.002` budget and a fixed `allow_fp16_reduced_precision_reduction = False` (which forces FP32 GEMM reduction, entirely avoiding the FP16-accumulate tier). 

Because the tightest accuracy margin on the baseline stack sits at `max_abs = 0.00195` (97.5% of budget) and `0.00211` (which clears via the relative tolerance arm), the accept/reject rules for these implementations are exceptionally strict.

| ID | Optimization | Precision Class |
|---|---|---|
| **G0.1c** | SDPA replaces the manual `matmul→mask→softmax→matmul` loop (`is_causal=True`) | Exact | 
| **G0.2c** | Fused QKV: one `[d, 3d]` GEMM for three `[d, d]` | Exact | 
| **G0.3** | Strided head views into SDPA — no `.contiguous()` copy | Exact |
| **G0.5** | All-ones-mask fast path — skips no-op `masked_fill` passes | Exact |
| **G1.1c** | LayerNorm affine folded into consumer GEMM weights; LN becomes pure reduction | Exact (bit-identical) |
| **G1.2** | Attention scale `head_dim^-0.5 = 2^-3` folded into `W_Q` | Exact (power-of-two) |
| **G6.4bc** | FP16 storage for Q/K/V/out_proj around SDPA; unlocks automatic flash/mem-efficient dispatch | precision-reducing (FP32 accum) |
| **G6.4a_v2c**| FP16 storage for the `ffn_in` GEMM; casts back to FP32 immediately | Precision-reducing |
| **G2.4(b)**| `torch.compile(mode="reduce-overhead")` → CUDA-graph replay | Exact (no value change)|
| **G4.7c** | Fused `ffn_in` GEMM + exact-erf GELU epilogue on warp-specialised `mma.sync` kernel | Precision-neutral |
| **G4.3** | Warp-specialised `mma.sync` GEMM + CUTLASS-grade epilogue for attention projections | FP32-accumulate |
| **G6.6** | cuBLASLt explicit algorithm search + fused bias epilogue for FFN GEMMs | N/A |
| **G7.0** | Sequence-chunked causal forward — query chunks vs an incremental K/V cache; makes `S = 100000` runnable at all | Exact (chunking drops no term) |
| **G7.1** | Chunking gate restated in bytes against real VRAM, replacing a fixed `B·S·d` element proxy | Exact (routing only) |
| **G7.4** | One bottom-right-causal `_flash_attention_forward` call per chunk — deletes the two-block split, the FP32 LSE merge and the `.contiguous()` transpose copy | Precision-neutral (accuracy improved) |
| **G7.5** | LayerNorm applied directly to the FP16 residual — CUDA already accumulates in FP32 for half input | Exact (bit-identical) |
| **G7.6** | `torch.compile` on the chunk body (`CHUNK_COMPILE` default on) | Exact (no value change) |

### Hardware Limit Reached
The accuracy-legal roofline is `12·M·d²·L / 165.2e12` (GEMM FLOPs at the
FP16-storage/FP32-accumulate tier) + measured SDPA + a 0.855 µs/kernel body
floor. Per representative shape:

![roofline proximity](assets/roofline.svg)

| Regime | Shapes | Binding wall | Shipped vs roofline |
|---|---|---|---|
| Latency / launch | 2–4, 12 | Kernel-body floor + Tensor-core pipeline fill on sub-roofline GEMMs | At the wall, without Hopper style Megakernel available (RTX 4090 runs Ada) |
| Compute | 8 | 165.2 TFLOP/s (FP16 Tensor + FP32 Accumulate)  | GEMMs at **94–96%** isolated, 84% in-model (L2 contention across 34 kernels) |
| Memory bandwidth | 6 | DRAM GB/s | Elementwise at the roofline — **23.6 GB predicted vs 22.7 GB measured** |
| O(S²) attention + memory | 13 | Flash `O(S²)` + LayerNorm/residual traffic | SDPA at its achievable floor, LN traffic runs against L2 |

For more information, [`docs/PARETO_FRONTIER_ANALYSIS.md`](docs/PARETO_FRONTIER_ANALYSIS.md).


### Rejected Optimizations

Most optimizations failed either because they did not meet the accuracy requirements, or provided marginal gains proportional to the precision reduction caused.

| Attempt | Mechanism | Rejection |
|---|---|---|
| **Inline-PTX `mma.sync` FP16 GEMM** (G4.4) | Raw PTX targeting the FP16-accumulate tier (unavailable via cuBLAS on Ada). | **Performance (< 2%).** Only hit 55% of its tier's potential. The one shape it helps already spends ~90% of its budget on FP16 *storage*. |
| **Warp-specialised `mma.sync` pipeline** (G4.3 / G4.7) | Decoupled producer/consumer warp groups with zero-extra-shared-memory epilogue. | **Accuracy / HW limits.** FP16 arm failed causal bounds by 5× (0.0055). FP32 shipped, but only benefits massive shapes due to Ada's half-rate FP32. |
| **SASS-level rewrite** (G4.5) | Disassembling and reordering memory instructions for optimal scheduling. | **Toolchain wall.** CUDA 13.1 `nvdisasm` lossy-renders ELF sections, making reassembly impossible. |
| **CUTLASS bake-off** (G4.6) | Sweeping newer CUTLASS FP16-accumulate templates across configs. | **Accuracy / Perf limits.** Failed 80% perf gate, then permanently closed on accuracy (FP16 at K=512 costs ~7.5× error; 6 of 8 shapes fail). |
| **Fused causal megakernel** (G5.MEGA) | Fusing 4 layers into one block to keep residuals on-chip and avoid HBM traffic. | **Slower (×0.74).** Shared memory limits (96 KB) restricted it to 1 block/SM, killing occupancy and latency hiding. |
| **Offline cuBLASLt algo selection** (G6.9) | Static profiling and pre-selection of the best cuBLASLt algorithms per shape. | **Negative yield.** 25 of 27 signatures showed < 2% win; single isolated win degraded end-to-end performance by 12.5%. |
| **BF16 whole model** | Global precision reduction to BF16. | **Accuracy (~5.5× over budget).** Reached `0.0110` max error; 13.6% of elements fail on the first shape. |
| **BF16 FFN only** | Precision reduction isolated to FFN layers. | **Accuracy (~5.6× over budget).** Same order of error as whole-model, ruling out "attention was the risk". |
| **FP8 FFN** | Per-channel scaled FP8 quantization. | **Accuracy (~33× over budget).** 20 out of 20 seeds failed the accuracy check. |
| **INT8 FFN** | Fixed step-size quantization. | **Accuracy (~15× over budget).** Fixed step size cannot auto-range O(1) activations. |
| **Split-precision FP8** | Ootomo-style mixed FP8 precision. | **Arithmetic ceiling.** Passes accuracy at `k=4`, but yields no speed win. A 4-GEMM kernel's ideal 82.6 TFLOP/s matches TF32's existing peak. |
| **`torch.compile(max-autotune)`** | Automated graph compilation and tuning. | **Accuracy (~2.2–2.4× over budget).** Compilation incorrectly picks Triton software-TF32 over cuBLAS native. |
| **Degree-7 minimax GELU polynomial** | Polynomial approximation for GELU activation. | **Accuracy (~84× over budget).** The catalogued ~1e-6 estimate was wrong by 5 orders of magnitude (caught without a GPU). |
| **Row-14 `chunk_q` retune** (G7 step 5) | Re-deriving the adaptive query-chunk sizer to use the memory freed by G7.4. | **No prize.** 1024 / 2048 / 3072 measure 9653 / 9655 / 9627 ms — flat to 0.3%. The chunk loop was never chunk-count-bound. |
| **Second CUDA stream for chunk overlap** (G7 step 7) | Overlapping chunk *i+1*'s projections with chunk *i*'s attention. | **Nothing to overlap.** The path is ~97% GPU-busy in steady state; closed on the profile without being built. |
| **Multi-head Latent Attention** (G8.0) | DeepSeek-style KV compression: cache one low-rank latent instead of K and V, absorb the up-projections into `W_Q`/`W_O`. | **Accuracy (500–1000× over budget).** K and V are independent full-rank weights, so the exact latent is `d_c = d` — no compression at all — and the measured singular spectrum is flat (rank `d/2` keeps 79% of energy at every `d`). Every smaller rank lands at `max_abs` 1.0–2.3. MLA also shrinks a *decode* KV cache; this harness is prefill-only. |
| **FlashAttention-3** (G8.2) | Hopper-generation attention: `wgmma` async tensor cores, TMA bulk copies, warp-specialised GEMM/softmax overlap. | **Wrong hardware.** FA3 requires `sm_90a`; this is an Ada `sm_89` card with no TMA and no `wgmma` — the same wall that disqualified ThunderKittens. PyTorch already dispatches its vendored FA2 kernel. Ceiling measured anyway: attention is 7.6–38.2% of the forward on rows 1–13, so even FA3's claimed 1.5–2.0× would cap the whole-model gain at 2.5–19.1% *if* it ran. |
| **FP16-accumulate GEMM + FP32 carry** (G8.2) | Run the tensor core at the un-throttled 330 TF FP16-accumulate rate, promote to an FP32 carry every few columns of K to rebuild precision outside it. | **Not accumulate-bound.** Accuracy *is* reachable at the official `K=128` with the finest carry (5 of 12 rows under the 0.00180 ship ceiling) — but these GEMMs achieve only 0.6–27% of even the 165.2 TF lower tier, and FP32-accum vs FP16-accum differ by ~1% here, against ×1.43–1.53 at `K=512`. At `K=1024` the tier *is* real (CUTLASS FP16-accum hits 239.4 TF vs cuBLAS's 154.3, ×1.55) — but accuracy closes it: the fastest config lands at `max_abs` 7.39e-03, and adding the FP32 carry trades speed away faster than it buys accuracy, reaching 1.03× *slower* while still ~2× over the ship ceiling. An exact FP32 combine (the best of three rebuilds tried) gets closest — one row clears at 4 slices — but needs 32 slices at `K=1024`, by which point the speed is long gone. |
| **Grouped-query / multi-query attention** (G8.1) | Share one K/V across a group of heads. | **No exact form exists.** GQA/MQA need K/V shared across heads, but the baseline's per-head projections are independently initialised. Mean-pooling K/V per group measures `max_abs` 1.61–2.45, 87–91% of elements failing. |

> The chart below depicts the accuracy vs performance tradeoff of our various optimizations.

![speed vs accuracy](assets/accuracy.png)

---

## Agentic Auto Research

![agentic loop](assets/agentic.png)

* **Stage Based**. Escalation of optimization efforts based on stages, categorising different types of optimization and risk level.

* **Auto Research with Manual Input**. Facilitates wide exploration of optimization techniques.

* **Comprehensive Validation Loop**. Testing on official harness, with anti-gaming mechanisms to prevent AI from gaming the system.

* **Token Concious**. Single Sonet ochestrator summons subagents with appropriate models based on the task assigned.
---

## Performance - Accuracy Tradeoff
Optimizations are judged on the **performance-accuracy tradeoff**. Each optimization (whenever precision reducing), must provide substantial performance gains to warrant the precision reduction, as we have a fixed **accuracy budget**.


> $$\Delta EV = p_g G - p_f C$$
> * $\Delta EV$: The net expected value of shipping update X.
> * $p_g$: The probability that the observed gain is real (not just > noise or offline overfitting).
> * $G$: The magnitude of the gain (e.g., improved accuracy).
> * $p_f$: The probability of failure (encountering an "unseen shape" that breaks the system).
> * $C$: The cost of failure (your current total score, since exceeding the budget drops the score to 0).

A strict ceiling of `max_abs ≤ 0.00180` (90% of budget) and reserve target
`≤ 0.00170` is applied. Minimum-gain floor **~0.3–0.5% whole-model** for anything with precission reduction. **Exact transforms (the G1 folds, CUDA graphs, the G4.7 exact-erf epilogue) cost
zero budget**, and is accepted whenever it provides a performance gain.

---

## Reproducability

### Prerequisites

**Development Environment**
* Ubuntu Server LTS 26.04 (Linux 7.0 Kernel) with Slurm and Apptainer
* NVIDIA GeForce RTX 4090 (`sm_89 Ada Lovelace`), 24564MiB VRAM
* 32GiB DDR5 Ram
* ~15GB Disk Space required for full Image

### Run

> The evaluation runs `torch_transformer_benchmark.py` — the official harness
> with our `UserOptimizedTransformer` in place of the stub. **It is the single
> source of truth**; there is no separate development copy to drift from it, so
> what is scored is exactly what a grader executes.
>
> Our contribution is fenced by `>>> BEGIN user ... >>>` sentinels and
> everything outside them is byte-for-byte the reference harness. Two
> independent guards hold that: `tools/verify_baseline.py` diffs all 20 frozen
> symbols against the judges' own script at AST level, and
> `tools/sync_entrypoint.py --check` asserts the file is still exactly
> *(canonical harness + our sentinel blocks)*. `run_eval.sh` and the Slurm
> sweep both refuse to run if either fails.

```bash
# 1. Build the reproducible image  (-> /scratch/kernel.sif; ~10 min)
bash infra/apptainer/build.sh

# 2. Run the official evaluation  (rows 1-13; row 14 is unscorable by the harness)
./run_eval.sh                       # or: make eval
#    per-row logs + a summary table land in results/logs/
#    + row-14 chunked-capability probe:  RUN_ROW14=1 ./run_eval.sh

# 3. Regenerate the README figures
make figures                        # tools/make_figures.py -> assets/*.svg

# 4. Integrity checks
make check                          # verify_baseline + sync_entrypoint --check + check_validity
bash infra/verify_submission.sh dist/techjam2_*.tar.gz    # after `make package`

# 5. Re-splice our model onto a freshly published canonical harness
make entrypoint                     # self-hosting; only needed if the judges
                                    # update torch_transformer_benchmark.py
```

### Tools, Libraries, Datasets
- **Development Tools:** Claude Code · Slurm · Apptainer · NVIDIA Nsight Compute / Systems 13.1 ·
  `nvcc` 13.1 · git · tmux.
- **APIs:** None.
- **Libraries / Frameworks:** PyTorch 2.13.0 (cu130) · Triton 3.7.1 · CUDA /
  inline PTX · CUTLASS 4.7.1 (header-only) · cuBLAS / cuBLASLt · pybind11 ·
  NumPy · `torch.compile` (Inductor).
- **Datasets / Assets:** None. Inputs are synthetic random tensors from `generate_random_case`.

### Repository
```
├── torch_transformer_benchmark.py   THE artefact — official harness + our model,
│                                    single source of truth. Edit only inside the
│                                    `>>> BEGIN user ... >>>` sentinels.
├── run_eval.sh · Makefile           Standardized eval
├── csrc/                            Hand-written CUDA / C++ / inline-PTX (csrc/README.md — 3 files build at eval)
├── tools/                           verify_baseline · sync_entrypoint · check_validity · archive · make_figures · parse_ncu · slurm
├── experiments/                     70+ g0-g7 investigation drivers (experiments/README.md)
│   └── g7_0_row14_golden.json       Committed Row-14 reference fingerprint (B=32) — check 6
├── infra/
│   ├── apptainer/                   kernel.def + build.sh — Reproducible image
│   ├── slurm/                       *.sbatch — Batch scripts
│   └── run_container.sh · package.sh · verify_submission.sh
├── results/
│   ├── logs/                        120+ Slurm job receipts — every number in the docs traces here
│   └── artifacts/                   NCU JSON summaries, ground_truth.csv
├── archive/                         MAP-Elites elite-config store (archive/README.md)
├── assets/                          README figures (Generated via make figures)
├── docs/                            ARCHITECTURE · PROGRESS (55-step log) · DOCUMENTATION · FINAL_SCORECARD
│                                    · PARETO_FRONTIER_ANALYSIS · ACCURACY_BUDGET · SETUP · CATALOGUE · AGENTS · ...
└── CLAUDE.md                        The agent's operating manual
```

---

## Limitations and Future Works

- **Forward pass only.** No backward / training path.
- **Row 14 (`S = 100000`) is unscorable by the provided harness.** A chunked
  *baseline* for a genuinely scored comparison would be helpful.
- **Ada-specific.** The precision choices, tile sizes, and the "no megakernel"
  conclusion are calibrated to `sm_89`; a Hopper port would reopen TMA +
  `wgmma` + a persistent fused kernel and change the answer.
- **The last ~10–15% to roofline is left on the table** — Recovering it means CUTLASS-grade / persistent-kernel engineering that Ada cannot achieve on standard toolchains.
- **The accept/reject heuristic is tuned to this exact budget** (`0.002 /
  0.02`), a tighter budget would retroactively un-ship the two FP16 casts.
- **Support for Autoregressive Generation**. Expand to optimize KV-cache techniques if harness requires autoregressive generation instead of full sequence processing.
  
---

## License · Acknowledgments

[MIT](LICENSE) © 2026 Yeo Shu Yi.

Built with [Claude Code](https://claude.com/claude-code). Frozen baseline and
evaluation harness per the TikTok TechJam problem statement (Problem 3 —
*Implement a GPU Kernel for a Transformer Layer*).
