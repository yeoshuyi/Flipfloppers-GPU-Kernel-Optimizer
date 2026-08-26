# Optimisation Catalogue

Read before proposing. Pick **one** per iteration. Cite the profiler fact.
Do not go outside this list without stating why it is insufficient.

**Order:** `G0 → G1 → G2 → G4.0 → G3 → G4.2 → G4.1 → G4.3 → G4.4/G4.5`

> **Track A — the safety net.** G0 + G1 + G2 only. One to two days, entirely
> conventional, expect ~5–8x. Build it FIRST and verify it passes the full
> sweep. It exists so that a megakernel which does not converge by the deadline
> still leaves you with a valid submission. Everything from G4.0 onward is
> Track B and may be abandoned at any point without losing the run.

---

## G0 — Structural. Apply unconditionally, all regimes.

| # | Optimisation | Gain | Notes |
|---|---|---|---|
| G0.1 | SDPA with **`is_causal=True`** | 1.5–3x | Explicit `attn_mask` forces SDPA off the flash backend onto the slow math path |
| G0.2 | Fused QKV: three `[d,d]` → one `[d,3d]` | 1.1–1.2x | Plain attributes only |
| G0.3 | Kill `_split_heads` `.contiguous()`; write GEMM into `[B,H,S,D]` | 1.2–1.4x | Baseline burns ~96 MB/forward here |
| G0.4 | Cache causal mask by `seq_len` | free | Baseline rebuilds it in every layer, every call |
| G0.5 | All-ones-mask fast path | 1.05–1.15x | See CLAUDE.md dispatch |
| G0.6 | 128-bit vector loads (`float4`) | 1.05–1.1x | `LDG.128` vs 4x `LDG.32` |

G0 removes ~336 MB/forward of zero-arithmetic traffic (72 MB qkv transposes +
24 MB context transpose + 240 MB score round-trips) ≈ 333 µs against a 487 µs
TF32 compute floor.

---

## G1 — Constant folding. Exact, zero accuracy cost. **Prerequisite for G4.**

| # | Optimisation | Effect |
|---|---|---|
| G1.1 | `W' = W·diag(γ)`, `b' = Wβ + b` | LayerNorm affine **vanishes**; LN → pure reduction. Apply `norm1→{q,k,v}`, `norm2→ffn_in`. Not `final_norm` (no consumer). |
| G1.2 | `W_Q *= d_k^(−1/2)` | Removes elementwise multiply over all `[B,H,S,S]`. Pass `scale=1.0` to SDPA. |
| G1.3 | `W_Q *= log2(e)` | Softmax → raw `exp2` → one `MUFU.EX2`. **Only if you own the softmax.** SDPA owns its own exponential. |
| G1.4 | Pre-swizzle to `ldmatrix.m8n8.x4` order | Zero runtime permutation, no dequant writeback |
| G1.5 | Per-**channel** quant scales | Per-tensor exceeds the 1% budget. Free — weights frozen. |
| G1.6 | One contiguous weight arena | Single `accessPolicyWindow` covers the model |

All run **once**, lazily on first forward, after `load_state_dict`,
`.to(device,dtype)`, `.eval()`. **Verify `max_abs` unchanged after each.**

> A megakernel cannot fold affines at runtime, swizzle per iteration, or compute
> scales on the fly. G1 is a prerequisite of G4, not a stepping stone to it.

---

## G2 — Precision and residency

| # | Optimisation | Gain | Notes |
|---|---|---|---|
| G2.1 | BF16 GEMMs, **FP32 residual stream** | ~2x | Cast at GEMM inputs only → one rounding per layer, not compounded through the residual |
| G2.2 | FP32 softmax + LayerNorm | — | Match the reference exactly |
| G2.3 | L2 persistence on the arena | 1.1x default, **1.5x+ tiny** | — |
| G2.4 | CUDA Graphs — `torch.compile(mode="reduce-overhead")` | 1.2x default, **3x+ tiny** | Compile **lazily on first forward**; never rely on `--compile-user`, graders may not pass it |
| G2.5 | Cast output back to FP32 | — | Else output is quantised at 0.4% against a 1% bound |
| G2.6 | **FP8 e4m3 FFN**, per-channel scales | ~1.5x overall | 65% of FLOPs. Error averages down: `eps/sqrt(K)=6%/45.3~0.14%`. **Never in attention.** Gated on the Phase-0 probe. |
| G2.7 | INT8 FFN, per-channel — *alternative to G2.6* | ~1.5x | Same 330 TOPS tier as FP8. Use only if the Phase-0 probe shows FP8 unavailable in this torch/triton build. GELU outputs bounded below by −0.17, so range is tame. |
| G2.8 | Split-precision `A = A_hi + A_lo` (both BF16) | ~FP32 accuracy | `A_hi·B_hi + A_hi·B_lo + A_lo·B_hi`. 3 BF16 matmuls at 2x FP32-CUDA-core throughput. **The rung between "BF16 works" and "FP8 fails"** — Ootomo & Yokota (2022). |

```cpp
size_t sz = 20u << 20;
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, sz);
attr.accessPolicyWindow = {weight_arena, sz, 1.0f,
                           cudaAccessPropertyPersisting,
                           cudaAccessPropertyStreaming};
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

---

## G3 — Fusion and layout

| # | Optimisation | Gain | Notes |
|---|---|---|---|
| G3.1 | Fused FFN tile — intermediate never hits HBM | 1.2–1.4x | 64 tok × 512 hidden = 64 KB, GELU in-register, contract against `ffn_out` slice. Token-parallel ⇒ **no grid sync**. Saves ~48 MB/forward. |
| G3.2 | Fused LayerNorm + residual | 1.05–1.15x | **Profile first** — Inductor may already do it |
| G3.3 | Warp-shuffle reductions | 1.05–1.1x | `__shfl_xor_sync` butterfly, 5 steps, no shared traffic. Welford for one-pass variance. |
| G3.4 | XOR swizzle `col ^ (row % 8)` | 1.1–1.2x | Conflict-free without padding. **Do not pad** — breaks 128-bit alignment. |
| G3.5 | `cp.async` multi-stage pipeline | 1.15–1.3x | No TMA on Ada — hand-rolled |
| G3.6 | Minimax deg-7 GELU | 1.02–1.05x | `x<−5→0`, `x>5→x` exact; polynomial between, ~1e-6. Reference is `approximate="none"`. **Measure against `erff` first.** |
| G3.7 | `__launch_bounds__` register sweep | 1.05–1.2x | Compiler default is often wrong for fused kernels |

---

## G4 — Megakernel. See `docs/MEGAKERNEL.md` before touching any of these.

| # | Optimisation | Gain | Notes |
|---|---|---|---|
| G4.0 | **Two-kernel form** — attn block + FFN block, 12 launches in one graph | 1.3–1.8x | Same tiling, `grid.sync()` deferred. **Build first.** On the critical path, not a detour. |
| G4.1 | Persistent megakernel, cooperative launch | 1.3–2x tiny | Residual stays in FP32 registers across layers |
| G4.2 | K-dimension splitting | enables depth | Halves per-iteration shared. Before G4.3. |
| G4.3 | Warp specialisation (Loader/Consumer/Storer/Controller) | 1.2–1.4x | Tuned point: Consumer 16→8, stages 2→4 |
| G4.4 | `mma.sync` with FP16 accumulate | up to 2x | 660 TFLOPS tier. Split-K: FP16 within 256-chunks, FP32 across. |
| G4.5 | Skip softmax max-subtraction | 1.05–1.1x | Post-G1.2 scores scale ~0.18x; overflow needs >128 ≈ 30σ. **Gate on `input_scale` + shape, keep the safe path.** |
