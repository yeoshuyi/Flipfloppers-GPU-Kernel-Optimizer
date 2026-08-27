# RESUME — live cursor for the continuous optimization loop

Plan: `~/.claude/plans/crispy-cooking-pine.md`
Target: **CLAUDE.md official 14-row causal matrix** — now specifically **official row 6**
(B=10000, d128, h4, S128, L4, ffn128, causal; tok 1.28M).
Constraints: precision-neutral (match the shipped `_optimized_forward_causal` precision exactly);
sbatch only; no git remote (local commits ARE the deliverable); document while acting; commit per unit.

---

## NOW

- **Iteration:** 7 — **G5.MEGA**: per-sequence fused megakernel for official row 6 (user: "try 1 then 2")
- **Phase:** 1 — v2 (mma.sync GEMMs) built; correctness + speed job in flight.
- **v1 (scalar): CORRECT (g5_5 run158 PASS) but x0.227 (231ms vs 52ms).** Needed tensor cores.
- **v2 (this):** `csrc/g5_mega_causal.cu` rewritten — `gemm_rowreg` mma.sync m16n8k16 f32-accum for
  qkv/out_proj/ffn_in (4-warp M-split, one n8-tile at a time so acc = 8 floats/thread, A from shared,
  B streamed from L2). `xr` residual row in registers. sK/sV fp16 shared (64KB). qkv output
  round-trips a `[B][SEQ][3D]` fp16 global scratch (d128 shared can't hold n1+q+k+v). Attention /
  LN / GELU / ffn_out still scalar. Committed `<hash>`.
- **Next concrete action:** read `results/g5_6_mega_speed_run<J>.log`.
  - compile error / ptxas fail → fix .cu.
  - g5_5 OVERALL PASS + g5_6 shows mega ≥ x1 → good; if ≥ x1.5 go to Phase 2 integration.
  - g5_5 FAIL (mma fragment bug) → debug: dump one GEMM's output vs a torch reference in the probe.
    Likely: mma A/B fragment layout, the `__pragma unroll 1` n-tile loop, or the global-scratch
    read-after-write fence.
  - g5_6 still slow → attention scalar loop is likely next (profile which stage).
- **Phase 0 result (g5_5 run157):** the scalar megakernel is CORRECT — max|mega - fp64| =
  1.05-1.39e-3 across 5 trials, `fail 0`, passes the 0.002 budget with MORE headroom than the
  shipped path (1.28-1.49e-3). It does fp32 attention (vs the shipped fp16 flash), so it differs
  from the shipped path by ~1e-3 — a lateral/slightly-better rounding, not a bug.
- **Kernel compiled clean** (no spills reported). `csrc/g5_mega_causal.{cu,cpp}`: 128 threads/block
  = 1 per query row, scalar loops, shared `xs`[128][128]fp32 + `kh`/`vh`[128][32]fp16, `ctx`/`n1`/`n2`/`g`
  in registers, online softmax. Committed through `<hash>`.
- **In-flight:** g5_6 speed — `jobs/g5_6_mega_speed.sbatch` (runs g5_5 then g5_6) →
  `results/g5_6_mega_speed_run<J>.log`. Times the SCALAR megakernel vs the shipped compiled causal
  forward at B=10000 via CUDA-graph replay. ~15 min.
- **Next concrete action:** read the speed number.
  - mega already ≥ x1.5 shipped → skip most of Phase 1; go straight to integration (Phase 2).
  - mega x0.5-1.5 → Phase 1: replace the scalar GEMMs with `mma.sync` (the 3 fp16 GEMMs are
    128x128x128 / 128x384x128 -- warp-cooperative), keep attention scalar (tiny), re-measure.
  - mega << shipped (e.g. x0.1) → the scalar inner loops are the bottleneck; mma.sync + maybe
    a warp-per-query-row layout instead of thread-per-row. Iterate.
- **Phase 2 (after a speed win):** `_ensure_mega_plan` eager gate (hard row-6 specialist:
  causal ∧ d_model==128 ∧ num_heads==4 ∧ seq_len==128 ∧ num_layers==4 ∧ ffn_dim==128 ∧ tok≥2^19),
  OUT-PARAM custom op `g5::mega_causal_forward`, whole-body replacement in `_optimized_forward_causal`.
  Capture-verify (g5_3 pattern) + 40-trial ship-verify row 6 + controls.
- **THEN lever 2:** fused GELU→ffn_out→+residual kernel.
- **Why:** step 43 — row 6's 52.5ms forward is 38.9% LN/residual traffic + 8% standalone GELU. The
  fp32 residual `x` [1.28M,128] (655MB) round-trips HBM ~2 reads + 2 writes per layer × 4. A megakernel
  with **one CUDA block per sequence** (S=128, d=128 → one sequence fits on-chip) does all 4 layers
  with `x` resident in shared, zero residual HBM traffic.
- **Design (precision-faithful = precision-neutral by construction):** match every op to the shipped
  causal path exactly —
  - LN norm1/norm2 = pure `(x-mean)/rsqrt(var+eps)` (affine folded into weights), fp32; final_norm keeps affine.
  - qkv / out_proj / ffn_in GEMMs: fp16 storage, **fp32 accumulate** (like the shipped `F.linear(fp16)`).
  - attention: fp16 q/k/v, fp32 softmax WITH max-subtraction, fp16 probs, fp32 PV accumulate (match flash).
  - Q pre-scaled (folded into `_qkv_weight`), so scale=1.0. GELU: exact erf, fp32. ffn_out: fp32/TF32.
  - fp32 residual stream throughout.
  - Weights: the SAME folded fp16 tensors the shipped path builds (`_qkv_weight_fp16`, `_out_proj_weight_fp16`,
    `_ffn_in_weight_fp16`) + `layer.ffn_out.weight/bias` fp32 + `final_norm.weight/bias` fp32.
  - Shared budget (99KB): `x` [128,128] fp32 = 64KB persistent + ~32KB fp16 scratch (reused for
    n / qkv pieces / hidden). Attention tiled flash-style so scores never materialize [128,128].
- **Files:** `csrc/g5_mega_causal.cu` + `.cpp` (new). Custom op `g5::mega_causal_forward` OUT-PARAMETER
  (mutates the output tensor — steps 46-48 showed allocating ops don't survive cudagraph capture).
- **Next concrete action:** write `csrc/g5_mega_causal.{cu,cpp}` — a CORRECTNESS-FIRST version
  (can be slow: `x` in shared, scores materialized per-head if it fits, simple loops for GEMMs).
  Then `probes/g5_5_mega_correct.py`: build it, run 1 layer then 4 layers vs a pure-PyTorch replica
  of `_optimized_forward_causal`'s math, fp64-reference the pieces. Gate on bit-level-ish agreement
  before any speed work.
- **In-flight jobs:** none.
- **Pending decisions:** none yet — building.

## Build plan (G5.MEGA)

1. **Phase 0** — `csrc/g5_mega_causal.{cu,cpp}` correctness-first + `probes/g5_5_mega_correct.py` +
   `jobs/g5_5_mega_correct.sbatch`. Gate: matches the reference math at 1 and 4 layers, 0 failing
   elements vs the 0.002 budget on real row-6-shaped random input.
2. **Phase 1** — tune: shared-mem layout, mma.sync for the GEMMs, flash-tiled attention, occupancy.
   `probes/g5_5_mega_sweep.py` — best-of-5 CUDA-graph replay vs the shipped path at the row-6 shape.
3. **Phase 2** — integrate: `_ensure_mega_plan` eager gate (causal ∧ d_model≤128 ∧ S≤128 ∧ h==4 ∧
   L==4 ∧ tok≥2^19 — a hard row-6 specialist), wire into `_optimized_forward_causal` as a
   whole-body replacement when the gate fires. `probes/g5_3`-style capture verify (mega kernel symbol
   in the trace). 40-trial ship-verify row 6 + controls.
3b. **THEN lever 2** — fused `GELU→ffn_out→+residual` kernel (only after the megakernel resolves).
4. **Phase 3** — PROGRESS step 49, ACCURACY_BUDGET §8, SUBMISSION/DOCUMENTATION, archive, commit.

## Loop history (commits, newest first)

- `17154de` iter6/T3b: out-param op engages but 0% whole-model — reverted; official-matrix round closed (step 48)
- `4989375` iter5: G4.7 regime 1 (d512) verified engages through capture (step 47)
- `dd6a4ba` iter3 T2: d1024 GEMM warp-spec — negative (step 45)
- `b69c810` iter2 T1: SDPA + recompile audit — double negative (step 44)
- `84ed40a` iter1: profile official matrix (step 43)
- `12a68d4` iter0: doc refresh with G4.7 numbers
- `92bf628` (+`42cc466`/`a04bd31`) G4.7 ship (step 42) — verified valid (step 47)

## Cold-resume facts

- Shipped causal path = `_optimized_forward_causal` (benchmark.py); tags G0.1c/G1.1c/G6.4a_v2c/G0.2c/G6.4bc + G4.7c (d512 only). `_ensure_ffn_plan` gate == commit `09dee91`.
- G5.MEGA target = official row 6 ONLY: `TransformerConfig(batch_size=10000, seq_len=128, d_model=128, num_heads=4, ffn_dim=128, num_layers=4, causal=True)`.
- Folded weights the shipped path builds: `attn._qkv_weight_fp16` [384,128] (norm1 affine + head_dim**-0.5 scale + norm1 fold absorbed), `attn._qkv_bias_fp16` [384]; `attn._out_proj_weight_fp16` [128,128] = `out_proj.weight.to(fp16)` (NOT folded), `_out_proj_bias_fp16`; `layer._ffn_in_weight_fp16` [128,128] (norm2 affine folded), `_ffn_in_bias_fp16`; `layer.ffn_out` = original fp32 Linear; `self.final_norm` = original fp32 LayerNorm WITH affine. Builders: `_build_qkv_fold`, `_build_attn_fp16_fold`, `_build_ffn_in_fold`, `_ensure_folded_weights`.
- Reference math: read the `_optimized_forward_causal` body (benchmark.py ~L1075-1180) — LN(no affine)→n_fp16→qkv F.linear→split heads [B,4,128,32]→SDPA(is_causal,scale=1)→transpose/view→out_proj F.linear→+resid(fp32)→LN(no affine)→ffn_in F.linear→to fp32→gelu(erf)→ffn_out (fp32 Linear)→+resid. After 4 layers: final_norm (WITH affine).
- Budget atol 0.002 / rtol 0.02 disjunctive, `failed==0`. Row 6 shipped max_abs ≈ 0.00195 (97.5%).
- Verification method: capture-aware (`probes/g5_3_g47_capture_verify.py` pattern — force capture, census the kernel symbol). Isolated microbenches overstate wins (steps 45, 48) — require matched in-model BEFORE/AFTER.
- `tools/verify_baseline.py` + `tools/sync_entrypoint.py --check` before every commit.
