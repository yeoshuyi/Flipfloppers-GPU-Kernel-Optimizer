# RESUME — live cursor

Plan: `~/.claude/plans/crispy-cooking-pine.md`. No git remote — local commits ARE the deliverable.
Doc while acting; commit per unit. `tools/verify_baseline.py` + `tools/sync_entrypoint.py --check` before commits.

---

## NOW

- **Iteration 9 — G6.9 (offline cuBLASLt algo-selection): CLOSED, verdict `reject as marginal`.**
  PROGRESS step 50, commit `9b05d91`. benchmark.py UNCHANGED (step-42 / run142 state).
- **Step 51 (no code):** formal Pareto-frontier analysis delivered on user request —
  `docs/PARETO_FRONTIER_ANALYSIS.md` + artifact `f28d951c-f5b6-4165-b0b5-9304be667997`.
  Commit `2eeb964`. Establishes: 165.2 TFLOP/s is the accuracy-legal peak; row-6
  elementwise is at the BW roofline (23.6 GB predicted vs 22.7 measured); residual
  gaps are Ada-bound (no TMA/wgmma → no megakernel). Reinforces "loop has converged".
- **DONE — final scorecard delivered** (`docs/FINAL_SCORECARD.md`, PROGRESS step 52).
  Before/after: job 168 `results/official_causal_sweep_run168.log` (Σ 383.4→60.8 ms,
  6.3×, geomean 7.7×, all PASS). Per-stage + roofline: job 171
  `results/final_scorecard_run171.log` (`probes/final_scorecard.py`). Probe went
  167→169→170→171 (recompile-limit / UnboundLocalError / addmm-misbucket, each fixed).
  benchmark.py UNCHANGED. Nothing in flight.

### G6.9 outcome (for reference)
- Phase 1: 27 unique GEMM signatures (9 (M,d) × {qkv, proj, ffn_out}). G4.7c inert on all 14 shapes.
- Phase 2 (run164): 25/27 < 2%. Small-M sigs +0.00% (G6.6's K=512 split-K win does NOT reproduce at K=128).
  Only qkv M8192 d128 (+21% vs idx0) and qkv M8192 d1024 (+2.9%) cleared.
- Phase 2 step 5 (run165): d1024 = pure strawman (F.linear ≡ cuBLASLt-best, identical `ampere_s1688gemm_128x128`).
  d128 = real but ~3% KERNEL time only (11.89→11.50us), bit-identical, ~0.75% whole-model ceiling on shapes 1/9/10/11.
- Phase 3 (run166): routing qkv → cuBLASLt-best made the row-1 eager forward **−12.5%** (501.95→564.70us),
  output bit-identical. Per-call dispatch overhead swamps the 0.4us kernel saving. Same failure mode as
  G6.6c (reverted) and T3 (0% whole-model). → reject as marginal.

## Candidate queue (iteration 10+; all prior levers negative for the official matrix)

Steps 43–50 exhausted: SDPA/recompile (44), d1024 warp-spec (45), T3 fused ffn_in+GELU 0% (46-48),
G5.MEGA megakernel x0.74 parity (49), G6.9 cuBLASLt algo-selection reject-marginal (50).

Remaining un-run queue items from the plan:
1. **G4.5 — gated softmax max-subtraction skip.** Scores at ~0.125× post scale-fold; fp32 exp2 overflow
   needs ~30σ. Gate on `input_scale` + shape, keep max-subtracting fallback. Saves one score-matrix read
   pass — biggest on rows 13/6/8. Precision-neutral when gate holds. **NOT yet built.**
2. **Fused `ffn_out` for memory-bound d128 rows (6, 13)** — only if a profile shows the `[tok,128]` hidden
   round-trip is exposed. G5.MEGA already showed a per-seq megakernel is parity, but a narrower
   ffn_in→GELU→ffn_out→+residual (no attention) was NOT isolated. Marginal expected value.
3. Flash-decode-style split for the huge-token rows (6). Research-grade, unbuilt.

Honest state: the shipped causal path is at/near the hardware limit on every official shape. G4.5 is the
last catalogue lever with a credible precision-neutral payoff. If it also lands ≤0%, the loop has
converged — document convergence and stop.

## benchmark.py state

UNCHANGED from step-42 (run142). Shipped causal path = G0.1c/G1.1c/G6.4a_v2c/G0.2c/G6.4bc + G4.7c
(d512-only, inert on official matrix). `_ensure_ffn_plan` gate == commit `09dee91`.

## Loop history (commits, newest first)

- `9b05d91` iter9 G6.9 cuBLASLt algo-selection — reject as marginal (step 50)
- `8d4dd0a`/`e452811`/`072ac1b`/`c09236b`/`0a3ed42` iter9 G6.9 Phases 1-3 probes + runs
- `2cfd00a` iter8 G5.MEGA v3 CUTLASS-grade — x0.74 parity, not shipped (step 49)
- `17154de` iter6/T3b (48) · `4989375` iter5 (47) · `dd6a4ba` T2 (45) · `b69c810` T1 (44)
- `84ed40a` iter1 profile (43) · `12a68d4` doc refresh · `92bf628` G4.7 ship (step 42)

## Cold-resume facts

- cuBLASLt harnesses: `csrc/cublaslt_algo_fp16.cpp` (fp16/COMPUTE_32F; 7th arg `reduction_mask`) and
  `csrc/cublaslt_algo.cpp` (fp32/COMPUTE_32F_FAST_TF32; 6 args, no mask). Methods: `create_problem`,
  `num_algos`, `algo_info`, `run`/`lt_linear`, `time_algo`, `time_algo2`.
- Official rows: 1-5,9-12 d128/ffn128; 6 d128 tok1.28M; 7 d32/ffn32; 8 d1024/ffn1024; 13 d128 tok65536;
  14 d1024 (OOM baseline — no e2e). tok = B·S. Heads don't change GEMM signatures.
- Budget: atol 0.002 / rtol 0.02 disjunctive, `failed==0`.
- Docs current through PROGRESS step 50.
- Probes: `probes/g6_9_lt_official_sweep.py` (run164), `probes/g6_9b_lt_artefact_check.py` (run165),
  `probes/g6_9c_lt_e2e.py` (run166). Logs in `results/`.
