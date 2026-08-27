# RESUME — live cursor for the continuous optimization loop

Plan: `~/.claude/plans/crispy-cooking-pine.md`
Target: **CLAUDE.md official 14-row causal matrix ONLY** (ffn_dim ∈ {32,128,1024}).
Constraints: precision-neutral preferred; sbatch only; no git remote (local commits ARE the deliverable);
document while acting; commit after every meaningful unit; keep this file short + current-only.

---

## NOW

- **Iteration:** 6 / T3(b) — OUT-PARAMETER fused-FFN op for the memory-bound regime (official row 6)
- **Phase:** 40-trial ship-verify running (job **155**)
- **Story so far this iteration:**
  - g5_2 Phase 0 (run148): cfg 58 fused ffn_in+GELU is **x1.66 vs the shipped chain at d128/ffn128
    tok 1.28M** in isolation (memory-bound: the fused kernel runs in the standalone cast+GELU time).
  - The RETURNING op `g43::ffn_gelu_linear` silently fell back under CUDA-graph capture at d128
    (steps 46-47, confirmed by ws_gemm_kernel-census in g5_3 run153) — its internal 655MB fp32
    output alloc doesn't survive capture. d512 (64MB output) is fine.
  - **Fix (step 48, this iter):** `g43::ffn_gelu_linear_out` — out-param op (`mutates_args={"out"}`,
    no return) via the existing C++ `ws_gemm` out-param entry. `act` pre-allocated in the traced
    region so inductor's cudagraph tree owns it. `_ensure_ffn_plan` re-adds the `tok >= 2^19`
    regime + sets `self._ffn_membound`; causal FFN block branches on it. Regime 1 (d512) untouched.
  - **g5_3 capture-verify (run154): PASS** — `ws_gemm_kernel` now IN the d128 captured trace
    (x30 @ tok 1.05M, x60 @ row-6 shape L4). Replay diff vs fallback: d512 2.4e-7, d128 row6 2.6e-4
    (fp16-storage rounding — same precision tier as the shipped fp16 ffn_in; a lateral move).
- **In-flight:** job **155** = `jobs/g5_4_t3b_ship_verify.sbatch` → `results/g5_4_t3b_ship_verify_run155.log`.
  BOTH passes = working-tree benchmark.py, 40 trials; only `G4_7_FFN_CFG` differs (BEFORE=-1, AFTER=58)
  → clean isolation. Shapes: off_row6 (engages), off_row13/row1/row8 (gate off), int_large_batch_causal
  (regime-1 regression check), nc_large_batch. ~1.5h.
- **Next concrete action:** `cp /scratch/techjam2/runs/155.out results/g5_4_t3b_ship_verify_run155.log`;
  parse BEFORE(-1) vs AFTER(58) per shape.
  - **SHIP** iff: off_row6 all-40 PASS, `failed=0`, max_abs stays comfortably < 0.002 (BEFORE row6
    ≈ 0.00195 = 97.5% of budget — WATCH THIS), speedup ≥ +0.5%; every control unchanged within noise
    (esp. `int_large_batch_causal` = regime 1 must be identical).
    Then: PROGRESS step 48 (T3b ship), ACCURACY_BUDGET §8 row, refresh SUBMISSION/DOCUMENTATION
    (official row 6 optimized), regen `torch_transformer_benchmark.py` + `tools/verify_baseline.py`,
    git commit. No archive cell fits row 6's exact shape — document in PROGRESS/ACCURACY_BUDGET.
  - **NEGATIVE** (row 6 accuracy creeps over / regresses / < 0.5%): revert the T3b gate + wiring
    (keep the `ffn_gelu_linear_out` op def), regen entrypoint, PROGRESS step 48 negative. Then
    **declare the official-matrix precision-neutral optimization at END-STATE** — write a closing
    summary (T1/T2/T3 all explored), and pause the loop for user direction.
- **Pending decisions:** T3(b) ship vs official-matrix end-state — job 155 decides.

## Loop history (commits, newest first)

- `62d2041` iter6/T3b RESUME · `<t3b>` out-param op ENGAGES at d128 (g5_3 run154) · out-param op impl
- `4989375` iter5: G4.7 regime 1 verified engages through capture (step 47)
- `1e9debb` iter4 T3a: revert d128 mem-bound gate — silent fallback (step 46)  ← T3b re-adds it, cleanly
- `dd6a4ba` iter3 T2: FP32-accum warp-spec for d1024 GEMMs — negative (step 45)
- `b69c810` iter2 T1: SDPA backend + recompile audit — double negative (step 44)
- `84ed40a` iter1: profile official matrix — step 43
- `12a68d4` iter0: doc refresh with G4.7 numbers
- `92bf628` (+`42cc466`/`a04bd31`) G4.7 ship — verified valid (step 47)

## Findings (official-matrix optimization)

- T1 SDPA backend / recompile — closed negative (SDPA auto-picks FLASH; no recompiles).
- T2 d1024 GEMMs — closed negative (cuBLAS at 94-96% of FP32-accum roofline; warp-spec loses x0.87-0.92).
- **T3 d128 elementwise/GELU bar** (47% row 6, 42% row 13): fused ffn_in+GELU is x1.66 isolated at
  row 6; returning op fails capture (steps 46-47); **out-param op fixes it (step 48)** → job 155 verdict.
- Untried if T3b fails: fused GELU→ffn_out→+resid neutral kernel; token-parallel persistent megakernel
  for row 6's 38% LN/residual bar; fp16 both-FFN GEMMs [precision-reducing, dead vs row6's 97.5% budget].

## Cold-resume facts

- Shipped causal path = `_optimized_forward_causal`; tags G0.1c/G1.1c/G6.4a_v2c/G0.2c/G6.4bc + G4.7c
  (d512/ffn2048 regime 1, tok≥8192, VERIFIED engages). T3b adds a d128 tok≥2^19 regime — under verify.
- `_ensure_ffn_plan` (~L877): `big_ffn` (d≥512 ∧ ffn≥2048 ∧ tok≥8192) OR `membound` (tok≥2^19 ∧ ffn≥128);
  sets `_ffn_cur`=cfg and `_ffn_membound`. Traced FFN block: membound→out-param op, else→returning op.
- Budget atol 0.002 / rtol 0.02 disjunctive, `failed==0`. `tools/verify_baseline.py` guards the harness;
  `tools/sync_entrypoint.py` regen `torch_transformer_benchmark.py` before every commit.
- **Verification method**: `probes/g5_3_g47_capture_verify.py` (40 warmup → force capture → census
  for `ws_gemm_kernel` in the replay trace) is the REQUIRED gate for any fused-FFN change. The old
  `probes/g4_7_ffn_wiring_smoke.py` one-call check only proves gate+eager, NOT capture engagement.
- `_before_benchmark.py` = gitignored throwaway from ship sbatches.
- G4.7 kernel: `csrc/g4_4_warpspec_gemm.cu` cfg 51-76; cfg 58 = ACCF32 + fused erf-GELU.
- Official rows: 1-5,9-12 d128/ffn128 small; 6 d128 tok1.28M; 8 d1024/ffn1024; 13 d128 tok65536;
  7 d32/ffn32; 14 d1024 tok3.2M (OOM baseline, not run).
