# RESUME — live cursor for the continuous optimization loop

Plan: `~/.claude/plans/crispy-cooking-pine.md`
Target: **CLAUDE.md official 14-row causal matrix ONLY** (ffn_dim ∈ {32,128,1024}).
Constraints: precision-neutral preferred; sbatch only; no git remote (local commits ARE the deliverable);
document while acting; commit after every meaningful unit.

---

## NOW

- **Iteration:** 6 — resolve whether T3(a)'s step-46 revert was a FALSE ALARM
- **Phase:** g5_3 re-run with forced-d128 cases + census-based verdict (job in flight)
- **THE KEY UNCERTAINTY:** job 152 showed d512 has `ws_gemm_kernel` x30 in the captured trace
  (ENGAGED) but cfg58-vs-fallback replay diff = 2.4e-7 (fp32 epsilon — precision-neutral, NOT a
  fallback). My step-46 "silent fallback" call for T3/d128 was based on run150's "BEFORE≡AFTER
  bit-identical per-trial" — which is ALSO exactly what a precision-neutral engaged kernel produces.
  So step 46 may be WRONG and T3(a) (+5.7% on row 6) may actually be a valid ship.
- **Next concrete action:** read `results/g5_3_g47_capture_verify_run<J>.log`. The truth signal is
  `ws_gemm_kernel in cfg58 CAPTURED trace` for the FORCED-on d128 cases (tok 1.05M L2, tok 1.28M L4):
  - **`ws_gemm_kernel` PRESENT at d128** → step 46 was a false alarm. RE-APPLY the `tok>=2^19` gate
    regime (revert commit 1e9debb's benchmark.py change), re-run the 40-trial ship-verify to
    re-confirm row-6 speedup with a clean matched BEFORE/AFTER (BEFORE = a forced-fallback build via
    `G4_7_FFN_CFG=-1`, so speedup isn't confounded by 5-vs-40 trial), then ship: PROGRESS step 48,
    ACCURACY_BUDGET §8, commit. Update step 46 to "corrected".
  - **`ws_gemm_kernel` ABSENT at d128** → step 46 stands, T3(a) genuinely dead. Then iteration 7 =
    T3 route (b) out-param op / fused-ffn_out, OR declare end-state.
- **In-flight jobs:** g5_3 v3 — `squeue -u techjam2`; sbatch `jobs/g5_3_capture_verify.sbatch`;
  output → `results/g5_3_g47_capture_verify_run<J>.log`. ~15 min.
- **Pending decisions:** T3(a) ship vs dead — this job decides.
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
