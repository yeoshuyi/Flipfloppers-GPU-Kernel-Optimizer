# RESUME — live cursor

Plan: `~/.claude/plans/crispy-cooking-pine.md`. No git remote — local commits ARE the deliverable.
Doc while acting; commit per unit. `tools/verify_baseline.py` + `tools/sync_entrypoint.py --check` before commits.

---

## NOW

- **Iteration:** 9 — **G6.9: offline cuBLASLt algo-selection investigation** for the 14 official causal shapes.
  4-phase protocol (owner-specified): inventory → isolated search → e2e lookup → final validation.
- **Phase:** 1 (inventory) done in the probe header; **Phase 2 (isolated search) job in flight.**
- **benchmark.py state:** UNCHANGED from step-42 (run142). Do not modify the runtime model unless a
  candidate survives all 4 phases.

### Phase 1 inventory (analytic — probes/g6_9_lt_official_sweep.py header)

Causal forward GEMMs (G4.7c inert on every official shape: max ffn_dim 1024 < 2048):
`qkv` (M=tok, N=3d, K=d) fp16 BIAS · `out_proj`≡`ffn_in` (M, N=d, K=d) fp16 BIAS ·
`ffn_out` (M, N=d, K=ffn=d) **TF32** BIAS.  Row-major, TRANSA=T/TRANSB=N, sm_89, CUDA 13.0.
d==ffn for all 14 shapes ⇒ per (M,d) there are 3 signature types.
Unique (M,d) over shapes 1-13 (14 OOMs the baseline → no e2e): d128 M∈{128,512,2048,8192,16384,65536,1.28M};
d32 M=8192; d1024 M=8192.  **⇒ 9 (M,d) × 3 = 27 unique signatures.**

Prior coverage: **no exact match**. Patterns only — G6.6 (small-M FFN, K=512 → split-K win, bias-path
era); G6.7 (fp16 attention GEMMs, small M → default heuristic optimal, negative); G6.8 (ffn_in fp16
M=8192 → 0.36% whole-model, negative). Large-M (M≥8192) inherits G6.8's negative pattern; only the
small-M sigs (M∈{128,512,2048}, shapes 2/3/4/12) have a prior reason to expect a win — and at K=128
(not 512) the split-K lever is weak. Sweep checks all 27 anyway.

### Phase 2 method (artefact-free)

Best heuristic candidate (idx k) vs idx 0 (heuristic's own top pick), BOTH via the same
`cublasLtMatmul` `run()` — not vs PyTorch `F.linear` (different harness). fp16: `reduction_mask=2`
(policy — `allow_fp16_reduced_precision_reduction=False`); mask=7 informational only. tf32: unrestricted.
Best-improvement = MIN over 3 repeats. RETAIN only >2% reproducible at the admissible mask.

- **In-flight:** g6_9 — `jobs/g6_9_lt_sweep.sbatch` → `results/g6_9_lt_official_sweep_run<J>.log`. ~25 min.
- **Next concrete action:** read the log.
  - "PHASE 2 RESULT: no uncovered signature beats idx0 by >2%" → **STOP.** Document the negative
    (PROGRESS step 50), `no uncovered opportunity` / `reject as marginal`. Model untouched. Done.
  - Retained candidates listed → Phase 3: static `(sig)→algo` lookup in `benchmark.py` (NO search/
    calibration/sync/bench in the timed path), test the affected official shapes only. Then Phase 4:
    accuracy on all 14 + all-shape latency regression + confirm no unaffected-shape regression.
    Freeze the table; strip search machinery from timed execution.
- **Final verdict must be one of:** ship / promising but insufficient evidence / reject as marginal /
  reject as inaccurate / no uncovered opportunity.

## Prior session work (unchanged)

Shipped causal path = step-42 state (G0.1c/G1.1c/G6.4a_v2c/G0.2c/G6.4bc + G4.7c d512-only).
`_ensure_ffn_plan` gate == commit `09dee91`. Steps 43-49 all negative for the official matrix
(SDPA/recompile, d1024 warp-spec, T3 fused ffn_in+GELU 0% whole-model, G5.MEGA megakernel x0.74 parity).
Assets: `csrc/g5_mega_causal.*` (verified-correct megakernel ref), `g43::ffn_gelu_linear_out` (unused
cudagraph-safe op), `docs/ACCURACY_BUDGET.md` §8.

## Loop history (commits, newest first)

- `<g6_9>` iter9 G6.9: cuBLASLt algo-selection sweep Phase 1+2 (in flight)
- `2cfd00a` iter8 G5.MEGA v3 CUTLASS-grade — x0.74 parity, not shipped (step 49)
- `17154de` iter6/T3b (step 48) · `4989375` iter5 (step 47) · `dd6a4ba` T2 (45) · `b69c810` T1 (44)
- `84ed40a` iter1 profile (43) · `12a68d4` doc refresh · `92bf628` (+archive) G4.7 ship (step 42)

## Cold-resume facts

- cuBLASLt search harnesses: `csrc/cublaslt_algo_fp16.cpp` (fp16 storage / COMPUTE_32F; takes
  `reduction_mask` 7th arg) and `csrc/cublaslt_algo.cpp` (fp32 / COMPUTE_32F_FAST_TF32; 6 args, no mask).
  Both: `create_problem`, `num_algos`, `algo_info` (id/tile/stages/splitk/reduc/swizzle/ws/wave),
  `run`/`lt_linear`, `time_algo`, `time_algo2` ({gpu_ms, cpu_issue_ms}).
- Official rows: 1-5,9-12 d128/ffn128; 6 d128 tok1.28M; 7 d32/ffn32; 8 d1024/ffn1024; 13 d128 tok65536;
  14 d1024 (OOM baseline). tok = B·S. Heads don't change GEMM signatures.
- Budget atol 0.002 / rtol 0.02 disjunctive, `failed==0`.
- Docs current through PROGRESS step 49.
