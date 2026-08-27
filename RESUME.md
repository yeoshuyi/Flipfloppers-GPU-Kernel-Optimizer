# RESUME — live cursor for the continuous optimization loop

Plan: `~/.claude/plans/crispy-cooking-pine.md`
Target: **CLAUDE.md official 14-row causal matrix ONLY** (ffn_dim ∈ {32,128,1024}).
Constraints: precision-neutral preferred; sbatch only; no git remote (local commits ARE the deliverable);
document while acting; commit after every meaningful unit; keep this file short + current-only.

---

## NOW

- **Iteration:** 7 — LOOP PAUSED. Official-matrix precision-neutral round is at end-state.
- **Phase:** awaiting user direction.
- **Why paused:** PROGRESS steps 43-48 explored every cheap-to-moderate precision-neutral lever on
  the official 14-row causal matrix. **Nothing shipped.**
  - T1 (44): SDPA auto-picks FLASH already; no dynamo recompiles. Negative.
  - T2 (45): cuBLAS at 94-96% of the FP32-accum roofline on the d1024 GEMMs; warp-spec loses x0.87-0.92. Negative.
  - T3 (46-48): d128 fused ffn_in+GELU — returning op fails CUDA-graph capture; the **out-param op
    `g43::ffn_gelu_linear_out` engages through capture and is precision-neutral, but gives 0%
    whole-model** (g5_4 run155: row6 5.851x → 5.851x). The isolated g5_2 x1.66 was a microbench artefact.
  - Twice (45, 48) an isolated microbench overstated the win vs the matched in-model measurement.
- **benchmark.py state:** reverted to the run142-verified state — `_ensure_ffn_plan` gate predicate
  is **byte-identical to commit `09dee91`**. `g43::ffn_gelu_linear_out` is kept (registered, verified,
  unused) as an asset. Shipped behaviour unchanged since step 42.
- **Next concrete action (needs user input):** the remaining levers are large builds with untrustworthy
  projected payoff:
  1. **token-parallel persistent megakernel for official row 6** — targets the real bar (LN/residual
     traffic = 38% of row 6, step 43), keeping the [tok,128] residual in registers across 4 layers.
     Large; faces the same torch.compile/cudagraph integration wall as steps 46-48; payoff unmeasurable
     without building most of it.
  2. **fused GELU→ffn_out→+residual kernel** (G3.1 rebuilt on the warp-spec infra, precision-neutral) —
     same wall + the microbench-vs-in-model trust problem.
  3. **precision-reducing options** — fp16 both-FFN GEMMs at 0.002: dead vs row 6's 97.5%-of-budget max_abs.
  4. **stop** — the official matrix is at its precision-neutral end-state with the standard-kernel toolkit;
     shipped causal speedups (2.71x default / 7.66x tiny / 7.78x long-seq / 2.98x large-batch on the
     internal d512 sweep; the official rows carry the G0/G1/G6.4 chain) stand.
  Ask the user which of 1/2/4 to pursue, or whether to redirect (non-causal, a different regime, etc.).
- **In-flight jobs:** none.
- **Pending decisions:** iteration 7 direction — user call.

## Loop history (commits, newest first)

- `<step48>` iter6/T3b: out-param op engages but 0% whole-model — reverted; official-matrix round closed (step 48)
- `62d2041` / `<t3b>` iter6/T3b: out-param op `ffn_gelu_linear_out` — engages through capture (g5_3 run154)
- `4989375` iter5: G4.7 regime 1 (d512) verified engages through capture (step 47)
- `1e9debb` iter4 T3a: revert d128 gate — silent fallback (step 46)
- `dd6a4ba` iter3 T2: d1024 GEMM warp-spec — negative (step 45)
- `b69c810` iter2 T1: SDPA + recompile audit — double negative (step 44)
- `84ed40a` iter1: profile official matrix (step 43)
- `12a68d4` iter0: doc refresh with G4.7 numbers
- `92bf628` (+`42cc466`/`a04bd31`) G4.7 ship (step 42) — verified valid (step 47)

## Cold-resume facts

- Shipped causal path = `_optimized_forward_causal`; tags G0.1c/G1.1c/G6.4a_v2c/G0.2c/G6.4bc + G4.7c
  (d512/ffn2048 regime 1 only; verified engages through capture). No official-matrix optimization shipped.
- `_ensure_ffn_plan` gate == `09dee91`. `g43::ffn_gelu_linear` (returning, regime 1) +
  `g43::ffn_gelu_linear_out` (out-param, unused asset) both registered in `_ffn_register_op`.
- Budget atol 0.002 / rtol 0.02 disjunctive, `failed==0`. `tools/verify_baseline.py` guards the harness;
  `tools/sync_entrypoint.py` regen `torch_transformer_benchmark.py` before every commit (both currently clean).
- **Verification method for any fused-FFN change**: `probes/g5_3_g47_capture_verify.py` (force capture,
  census `ws_gemm_kernel` in the replay trace). The one-call `probes/g4_7_ffn_wiring_smoke.py` only
  proves gate+eager. And: **isolated GEMM/FFN microbenches overstate wins — require a matched in-model
  BEFORE/AFTER before shipping** (steps 45, 48).
- Official rows: 1-5,9-12 d128/ffn128 small; 6 d128 tok1.28M; 8 d1024/ffn1024; 13 d128 tok65536;
  7 d32/ffn32; 14 d1024 tok3.2M (OOM baseline, not run).
- Docs current through PROGRESS step 48; SUBMISSION/DOCUMENTATION/CLAUDE/CAUSAL_LEDGER carry G4.7 (step 42) numbers.
