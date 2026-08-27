# RESUME — live cursor for the continuous optimization loop

Plan: `~/.claude/plans/crispy-cooking-pine.md`
Target: **CLAUDE.md official 14-row causal matrix ONLY** (ffn_dim ∈ {32,128,1024}).
Constraints: precision-neutral preferred; sbatch only; no git remote (local commits ARE the deliverable);
document while acting; commit after every meaningful unit.

---

## NOW

- **Iteration:** 6 — T3(a) retry: cudagraph-safe OUT-PARAMETER custom op for the fused ffn_in+GELU
- **Phase:** implementing
- **iter 5 RESULT (step 47, commit `4989375`):** G4.7 regime 1 (d512) VERIFIED engages through
  capture — `ws_gemm_kernel` in the captured trace x30. G4.7 stands, no revert. The d128 fallback
  is diagnosed as the custom op allocating a 655MB fp32 output inside the cudagraph pool (×4 layers,
  B=10000) — `ffn_dim>=2048` doubles as an implicit output-size bound.
- **iter 6 hypothesis:** the fix is an OUT-PARAMETER op — `g43::ffn_gelu_linear_out(inp,w,bias,cfg,out)`
  with `mutates_args=("out",)`, `out` pre-allocated by the compiled graph (`torch.empty` inside the
  traced region, which inductor's cudagraph tree DOES handle, unlike an opaque op's internal alloc).
  The C++ out-param entry `ws_gemm(cfg,inp,w,bias,out)` ALREADY EXISTS in g4_4_warpspec_gemm.cpp.
- **Next concrete action:** add `g43::ffn_gelu_linear_out` to `_ffn_register_op` (mutates_args, no
  return, fake is a no-op); in `_optimized_forward_causal` when `ffn_cfg` set, do
  `act = torch.empty(*n2.shape[:-1], ffn_dim, dtype=fp32, device); torch.ops.g43.ffn_gelu_linear_out(
  n2_fp16, w, b, ffn_cfg, act)`. Re-add the `tok>=2^19` gate regime. Then:
  `probes/g5_3_g47_capture_verify.py` (add the d128 case with gate ON) → if `ws_gemm_kernel` appears
  in the d128 captured trace → 40-trial ship-verify rows 6/13/1/8 + controls → ship or negative.
- **In-flight jobs:** g5_3 re-run (weights-fix) — `results/g5_3_g47_capture_verify_run<J>.log`; the
  ws_gemm_kernel census in run151 was already conclusive (d512 engages).
- **Pending decisions:** if the out-param op ALSO silently falls back at d128 → declare the official
  matrix at its precision-neutral end-state (write a summary), stop the loop.
- **Why:** step 46 — the T3 d128 fused path SILENTLY FELL BACK under torch.compile(reduce-overhead)
  + CUDA-graph capture (ship-verify run150: row6 AFTER per-trial max_abs bit-identical to BEFORE).
  The wiring smoke missed it (calls opt(x) once = eager pre-capture warmup, never exercises replay).
  **G4.7 regime 1 (d512, step 42, SHIPPED) was smoke-checked the same inadequate way** — must confirm
  it genuinely engages through capture, else step 42 is bogus and reverts.
- **Next concrete action:** when job done → read `results/g5_3_g47_capture_verify_run<J>.log`.
  - d512 verdict "ENGAGED through capture" (ws_gemm_kernel in trace + cfg58-replay ≠ fallback-replay):
    G4.7 stands. Update PROGRESS/RESUME, commit. Then pick iteration 6 (T3 route (b) fused-ffn_out
    neutral kernel, OR the d128 elementwise bar via a non-op integration, OR declare end-state).
  - d512 verdict "SILENT FALLBACK": **serious** — revert G4.7 (step 42) wiring in benchmark.py,
    revert archive causal elites (42cc466/a04bd31 → back to g6_4bc 7.10x/2.66x), PROGRESS step 47
    documenting the reversal + the +9.5%/+12.1% being a measurement artefact, regen entrypoint,
    commit. Then reassess whether the custom-op-in-cudagraph approach is viable at all.
- **In-flight jobs:** g5_3 capture-verify — `squeue -u techjam2`; sbatch `jobs/g5_3_capture_verify.sbatch`;
  output → `results/g5_3_g47_capture_verify_run<J>.log`. ~15 min.
- **Pending decisions:** iteration 6 target — after g5_3.

## Loop history (commits, newest first)

- `1e9debb` iter4 T3a: revert d128 mem-bound gate — **silent fallback under capture** (step 46)
- `93e55b7` iter4 T3: RESUME (Phase-0 win at row 6 x1.66)
- `2d5c71a` iter4 T3: extend _ensure_ffn_plan (tok>=2^19) — later reverted
- `dd6a4ba` iter3 T2: FP32-accum warp-spec for d1024 GEMMs — **negative** (step 45); cuBLAS at 94-96% roofline
- `b69c810` iter2 T1: SDPA backend + recompile audit — **double negative** (step 44)
- `84ed40a` iter1: profile official matrix — step 43 (d128 elementwise/GELU is the bar)
- `12a68d4` iter0: doc refresh with G4.7 numbers
- `92bf628` (+ `42cc466`/`a04bd31` archive) G4.7 ship — **UNDER RE-VERIFICATION (iter 5)**

## Findings so far (official-matrix optimization)

- SDPA already optimal (FLASH auto-picked); no recompiles. — closed
- d1024 GEMMs: cuBLAS at 94-96% of FP32-accum roofline; warp-spec loses x0.87-0.92. — closed
- d128 rows: the fused ffn_in+GELU kernel wins x1.66 in ISOLATION at tok 1.28M but the
  custom op does not survive CUDA-graph capture at d128. — route (a) closed
- **OPEN: does G4.7's shipped d512 path survive capture?** (iter 5, in flight)
- Untried: T3 route (b) fused GELU→ffn_out→+resid neutral kernel; token-parallel persistent
  megakernel for row 6 (the real 38% LN/residual bar); fp16 both-FFN GEMMs at 0.002 [reducing].

## Key facts for a cold resume

- Shipped causal path = `_optimized_forward_causal` in benchmark.py; G0.1c/G1.1c/G6.4a_v2c/G0.2c/G6.4bc + G4.7c (d512/ffn2048, tok≥8192 — under re-verification).
- benchmark.py gate `_ensure_ffn_plan` (~L858) is byte-identical to commit 09dee91 after the T3 revert; docstring carries the step-46 negative.
- Official rows: 1-5,9-12 = d128/ffn128 small; 6 = d128 tok1.28M; 8 = d1024/ffn1024 tok8192; 13 = d128 tok65536; 7 = d32/ffn32; 14 = d1024 tok3.2M (OOM).
- Budget atol 0.002 / rtol 0.02 disjunctive, failed==0. `tools/verify_baseline.py` guards it. `tools/sync_entrypoint.py` regen `torch_transformer_benchmark.py` before each commit.
- **Wiring-smoke method bug**: `probes/g4_7_ffn_wiring_smoke.py` `run_case` calls opt(x) ONCE = eager warmup; does NOT exercise cudagraph replay. Any fused-FFN verification needs ≥20 warmup calls then replay-vs-fallback diff (that's what `probes/g5_3_g47_capture_verify.py` does).
- `_before_benchmark.py` regenerated by ship-verify sbatch via `git show <sha>:benchmark.py` (gitignored).
- G4.7 kernel = `csrc/g4_4_warpspec_gemm.cu` configs 51-76; cfg 58 = ACCF32 + fused erf-GELU, fp32 out. custom op `g43::ffn_gelu_linear` in benchmark.py `_ffn_register_op` (has a try/except that silently falls back to F.gelu(F.linear())).
