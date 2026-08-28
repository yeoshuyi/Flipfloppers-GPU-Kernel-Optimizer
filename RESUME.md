# RESUME — live cursor for the continuous optimization loop

Plan: `~/.claude/plans/crispy-cooking-pine.md`
Constraints: precision-neutral preferred; sbatch only; no git remote (local commits ARE the deliverable);
document while acting; commit per unit; keep this file short + current-only.

---

## NOW

- **Iteration:** 8 — **G5.MEGA v3 (CUTLASS-grade)**: user asked to attempt it after the v1/v2 negatives.
- **Phase:** v3 built; correctness (g5_5) + speed (g5_6) job in flight.
- **v3 design:** 256 threads/block (8 warps). Threads (2t, 2t+1) share token t; each owns HALF the
  residual row `xr[64]` fp32 IN REGISTERS (v2 spilled with xr[128]). x read once / written once,
  residual never round-trips. ALL 4 GEMMs tensor-core: qkv/out_proj/ffn_in = mma m16n8k16
  f16.f16.f32; **ffn_out = mma m16n8k8 tf32** (matches shipped fp32 nn.Linear @ high). qkv v-part
  uses `gemm16b` (batched acc) to dodge the n1-overwrite race. Cooperative LN via `__shfl_xor`.
  Attention: each thread does 2 heads of its token, online fp32 softmax. 3×[SEQ][D] fp16 shared (96KB).
- **Next concrete action:** read `results/g5_6_mega_speed_run<J>.log`.
  - ptxas: check registers / spill. If spilling badly → reduce `gemm16b<D>` acc (NT=16 → chunk it).
  - g5_5 FAIL → mma fragment / race bug. tf32 m16n8k8 layout is the newest risk (gemm_ffn_out).
    Debug: in g5_5, dump one layer's ffn_out output vs a torch reference.
  - g5_5 PASS + g5_6 shows x≥1 → progress; x≥2 → Phase 2 integration.
  - g5_6 still slow → the scalar attention inner loop (O(S²·HD) per token, 2 heads/thread) is the
    likely remaining bottleneck → mma-ify Q@K^T and P@V (head_dim=32, needs K/V transposed tiles).
- **benchmark.py state:** UNCHANGED from step-42 (run142). Nothing shipped for the official matrix.
  `_ensure_ffn_plan` gate == commit `09dee91`.

### What was explored this session (PROGRESS steps 43-49)

| # | target | result |
|---|---|---|
| 43 | profile official matrix | d128 rows: 38-47% LN/residual+GELU traffic; d1024 GEMM-bound; row13 SDPA 32% |
| 44 | T1: SDPA backend + dynamo recompile audit | **negative** — FLASH already auto-picked; 0 recompiles |
| 45 | T2: FP32-accum warp-spec for d1024 GEMMs | **negative** — cuBLAS at 94-96% of roofline; warp-spec x0.87-0.92 |
| 46-48 | T3: fused ffn_in+GELU on d128 rows | returning op fails cudagraph capture; **out-param op engages, precision-neutral, 0% whole-model** |
| 47 | re-verify G4.7 (step 42) engages through capture | **confirmed valid** — ws_gemm_kernel in the d512 captured trace |
| 49 | **lever 1 (G5.MEGA)** per-sequence megakernel for row 6 | **negative** — 2 correct prototypes (scalar x0.227, mma x0.127) both slower than cuBLAS+flash |
| 49 | lever 2 (fused ffn_out+GELU) | **not built** — structural mirror of T3, near-certain 0% |

### Reusable assets left behind

- `csrc/g5_mega_causal.{cu,cpp}` — a VERIFIED-CORRECT per-sequence fused causal megakernel
  (d128/h4/S128/L4). Slow (needs a CUTLASS-grade rewrite to win) but a working reference:
  mma.sync m16n8k16 f32-accum GEMMs, online-softmax attention, register-resident residual.
- `g43::ffn_gelu_linear_out` — out-param cudagraph-safe fused ffn_in+GELU op (registered, unused).
- `probes/g5_3_g47_capture_verify.py` — the required capture-aware verification for any fused-FFN change.
- `docs/ACCURACY_BUDGET.md` §8 — the ledger + the "isolated microbenches overstate wins" rule (steps 45, 48).

### If resuming — candidate directions (need owner input)

1. **CUTLASS-3.x-grade fused megakernel for row 6** — warp-specialised pipeline, flash-tiled
   attention in-kernel, zero spill. Large (20-40 iterations). The only path to move row 6.
2. **Non-causal regimes** — G4.3 (shipped non-causal) + G4.7's fused-GELU could extend; run132 showed
   x1.42-1.55 on non-causal d512/ffn2048 ffn_in. Only if non-causal is scored.
3. **Accept the current shipped model** — G0/G1/G6.4 causal chain + G4.7c (d512 sweep only). Done.

## Loop history (commits, newest first)

- `3cd186d` iter7: G5.MEGA lever 1 negative + lever 2 not built — official matrix end-state (step 49)
- `a96cc90`.. G5.MEGA v1 (scalar, correct, x0.227) + v2 (mma, correct, x0.127)
- `17154de` iter6/T3b: out-param op engages but 0% — reverted (step 48)
- `4989375` iter5: G4.7 regime 1 verified engages through capture (step 47)
- `dd6a4ba` iter3 T2 negative (step 45) · `b69c810` iter2 T1 negative (step 44) · `84ed40a` iter1 profile (step 43)
- `12a68d4` iter0 doc refresh · `92bf628` (+archive) G4.7 ship (step 42, verified)

## Cold-resume facts

- Shipped causal path = `_optimized_forward_causal`; G0.1c/G1.1c/G6.4a_v2c/G0.2c/G6.4bc + G4.7c (d512/ffn2048 only).
- Official rows: 1-5,9-12 d128/ffn128 small; 6 d128 tok1.28M; 8 d1024/ffn1024; 13 d128 tok65536; 7 d32/ffn32; 14 d1024 (OOM).
- Budget atol 0.002 / rtol 0.02 disjunctive, `failed==0`. `tools/verify_baseline.py` + `tools/sync_entrypoint.py --check` before every commit (both clean).
- G5.MEGA weights: stack per-layer `attn._qkv_weight_fp16`[384,128] / `_out_proj_weight_fp16`[128,128] /
  `_ffn_in_weight_fp16`[128,128] + `ffn_out.weight`fp32 + `final_norm.weight/bias`fp32. Builders:
  `_ensure_folded_weights` + `_build_{qkv,attn_fp16,ffn_in}_fold`.
- Docs current through PROGRESS step 49; SUBMISSION/DOCUMENTATION/CLAUDE carry G4.7 (step 42) numbers.
