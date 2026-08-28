# RESUME — live cursor

Plan: `~/.claude/plans/crispy-cooking-pine.md`. No git remote — local commits ARE the deliverable.
Doc while acting; commit per unit. `tools/verify_baseline.py` + `tools/sync_entrypoint.py --check` before commits.

---

## NOW

- **Iteration:** 8 CLOSED. **G5.MEGA (CUTLASS-grade megakernel) attempted — best x0.74 (parity),
  not a win. Not shipped.** benchmark.py unchanged from step-42 (run142).
- **Loop status:** the official 14-row causal matrix is at its optimisation end-state for what's
  proportionate here. Awaiting an owner direction call.

### G5.MEGA arc (PROGRESS step 49) — 4 prototypes, `csrc/g5_mega_causal.{cu,cpp}`

| version | correctness (vs fp64) | row-6 speed vs shipped 52ms |
|---|---|---|
| v1 scalar | PASS 1.05-1.39e-3 (tighter than shipped) | x0.227 (231ms) |
| v2 mma + fp16 shared k/v + qkv global scratch | PASS 1.15-1.44e-3 | x0.127 (413ms) |
| **v3 CUTLASS-grade** (256thr, residual 64+64 in regs = 0 spill, all-tensor-core, coop LN, online-softmax attn) | **FAIL** 1.7-2.2e-3 (over budget @ B=10000) | **x0.739 (71ms)** — fastest |
| v3 + fp32-scalar ffn_out + batched gemm | PASS 1.15-1.44e-3 | x0.171 (308ms) |

**Blockers to a win:** (1) `ffn_out` precision — shipped is effectively tf32x3 (`matmul_precision
"high"`); single-pass tf32 (fast) is ~0.5e-3 short → over budget; fp32-scalar is precise but ~80 G
FMAs. Fix = tf32x3 + precomputed-GELU buffer (~1 iter). (2) **occupancy** — 96 KB shared ⇒ 1 block/SM
⇒ no latency hiding; cuBLAS at 94-96% roofline (step 45) + optimal flash (step 44) leave no slack.
A real win needs the full CUTLASS-3.x machinery: `cp.async` weight pipeline + warp specialisation +
flash-tiled `mma` attention in-kernel. 15-30 more iterations.

`csrc/g5_mega_causal.cu` is left at the **verified-correct v3 variant** (batched gemm + fp32-scalar
`ffn_out`) as a working reference / starting point.

### If resuming G5.MEGA (the path to a win)

1. `ffn_out` tf32x3: after ffn_in, thread writes `gelu(hidden)` → an fp32 shared buffer ONCE
   (sA++sC = `sO`, 64KB); then 3 `mma.m16n8k8.tf32` passes (a_hi·w_hi + a_hi·w_lo + a_lo·w_hi),
   hi/lo split = `x - (x & 0xffffe000)`. Precomputing GELU also removes the per-element `erff`
   recompute (~5-10ms). Revert `gemm16` to per-n-tile (small acc, no spill) — keep `gemm16_batched`
   only for the qkv v-part.
2. occupancy: cut shared to ≤48KB (2 blocks/SM). Only 2 of the 3 `[SEQ][D]` fp16 buffers can stay;
   the qkv v-part or k/v may need a small global round-trip, OR warp-specialise so loader warps
   `cp.async`-prefetch weights while consumer warps `mma` — this is the G4.3 technique
   (`csrc/g4_4_warpspec_gemm.cu` has the named-barrier producer/consumer pattern to copy).
3. mma-ify attention (Q@K^T, P@V; head_dim=32) — replaces ~4ms scalar.

## What shipped this session: NOTHING for the official matrix

Shipped causal path unchanged from step 42 (G0.1c/G1.1c/G6.4a_v2c/G0.2c/G6.4bc + G4.7c d512-only).
`_ensure_ffn_plan` gate == commit `09dee91`.

### Explored + closed this session (PROGRESS 43-49)

- 43 profile · 44 SDPA/recompile audit (neg) · 45 d1024 warp-spec (neg, cuBLAS at roofline)
- 46-48 T3 fused ffn_in+GELU: out-param op engages through capture, precision-neutral, **0% whole-model**
- 47 G4.7 (step 42) re-verified — genuinely engages through capture
- 49 G5.MEGA lever 1 (this) — parity, not a win; lever 2 (fused ffn_out) not built (T3 precedent = 0%)

### Assets left behind

- `csrc/g5_mega_causal.{cu,cpp}` + `probes/g5_5_mega_correct.py` + `probes/g5_6_mega_speed.py` —
  verified-correct megakernel reference.
- `g43::ffn_gelu_linear_out` — cudagraph-safe out-param fused ffn_in+GELU op (registered, unused).
- `probes/g5_3_g47_capture_verify.py` — required capture-aware check for any fused-FFN change.
- `docs/ACCURACY_BUDGET.md` §8 — ledger + "isolated microbenches overstate wins" rule (steps 45, 48).

## Loop history (commits, newest first)

- `2cfd00a` iter8: G5.MEGA v3 CUTLASS-grade — x0.74 parity, not shipped (step 49)
- `17154de` iter6/T3b: out-param op engages but 0% — reverted (step 48)
- `4989375` iter5: G4.7 regime 1 verified through capture (step 47)
- `dd6a4ba` T2 neg (45) · `b69c810` T1 neg (44) · `84ed40a` profile (43) · `12a68d4` doc refresh
- `92bf628` (+archive) G4.7 ship (step 42, verified)

## Cold-resume facts

- Official rows: 1-5,9-12 d128/ffn128 small; 6 d128 tok1.28M; 8 d1024/ffn1024; 13 d128 tok65536; 7 d32; 14 d1024 (OOM).
- Budget atol 0.002 / rtol 0.02 disjunctive, `failed==0`. Row 6 shipped max_abs ≈ 0.00195 (97.5%).
- G5.MEGA target/weights: `TransformerConfig(batch_size=10000, seq_len=128, d_model=128, num_heads=4,
  ffn_dim=128, num_layers=4, causal=True)`. Stack per-layer `attn._qkv_weight_fp16`[384,128] /
  `_out_proj_weight_fp16`[128,128] / `_ffn_in_weight_fp16`[128,128] + `ffn_out.weight`fp32 +
  `final_norm.weight/bias`fp32. Builders `_ensure_folded_weights` + `_build_{qkv,attn_fp16,ffn_in}_fold`.
- Isolated microbenches overstate wins — require matched in-model BEFORE/AFTER (steps 45, 48).
- Docs current through PROGRESS step 49; SUBMISSION/DOCUMENTATION/CLAUDE carry G4.7 (step 42).
