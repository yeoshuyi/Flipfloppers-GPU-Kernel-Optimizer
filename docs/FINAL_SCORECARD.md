# Final Scorecard — shipped vs baseline, official 14-row causal matrix

RTX 4090 (Ada, sm_89), CUDA 13.0 / PyTorch 2.13. Budget `atol 0.002 / rtol
0.02`, gate `failed == 0`. Row 14 (`S=100000`) OOMs the FP32 baseline — no
end-to-end path; 13 rows scored.

- **Before/after + speedup + accuracy:** `benchmark.py` per row, fresh process,
  `results/official_causal_sweep_run168.log` (job 168).
- **Per-stage split + roofline:** `probes/final_scorecard.py`,
  `results/final_scorecard_run171.log` (job 171). CUPTI census of the graphed
  forward, per-row clean `torch._dynamo` state. Census bucket-sum matches the
  `benchmark.py` median to 1–2 % on every row.

## 1. Before → after

| # | B | d | H | S | baseline | **shipped** | speedup | max_abs |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 64 | 128 | 4 | 128 | 1.0465 ms | **0.2109 ms** | 4.96× | 0.00137 |
| 2 | 1 | 128 | 4 | 128 | 1.0618 ms | **0.0778 ms** | 13.64× | 0.00137 |
| 3 | 4 | 128 | 4 | 128 | 1.0426 ms | **0.0881 ms** | 11.84× | 0.00137 |
| 4 | 16 | 128 | 4 | 128 | 1.0466 ms | **0.1116 ms** | 9.38× | 0.00137 |
| 5 | 128 | 128 | 4 | 128 | 1.7039 ms | **0.3727 ms** | 4.57× | 0.00137 |
| 6 | 10000 | 128 | 4 | 128 | 290.55 ms | **52.486 ms** | 5.54× | 0.00195 |
| 7 | 64 | 32 | 4 | 128 | 1.0209 ms | **0.1034 ms** | 9.87× | 0.00211 |
| 8 | 64 | 1024 | 4 | 128 | 8.3611 ms | **4.3271 ms** | 1.93× | 0.00141 |
| 9 | 64 | 128 | 1 | 128 | 0.9688 ms | **0.2099 ms** | 4.62× | 0.00145 |
| 10 | 64 | 128 | 2 | 128 | 1.0473 ms | **0.2130 ms** | 4.92× | 0.00138 |
| 11 | 64 | 128 | 16 | 128 | 4.3858 ms | **0.2857 ms** | 15.35× | 0.00137 |
| 12 | 64 | 128 | 4 | 32 | 1.0376 ms | **0.1229 ms** | 8.44× | 0.00141 |
| 13 | 64 | 128 | 4 | 1024 | 70.153 ms | **2.2088 ms** | 31.76× | 0.00137 |

Σ 13 rows: **383.4 ms → 60.8 ms (6.3×)**. Geometric-mean speedup **≈ 7.7×**.
All 13 PASS.

## 2. Per-stage latency (shipped, graphed forward, 4 layers)

`GEMM` = QKV + out_proj + ffn_in + ffn_out (cuBLAS/CUTLASS fp16, fp32 accum).
`SDPA` = flash attention (fp16, fp32 softmax). `GELU` = standalone erf-GELU
kernel. `LN+res` = fused LayerNorm ×2 + residual add ×2 + dtype casts (inductor
Triton).

| # | shipped | SDPA | GEMM | GELU | LN+res | dominant | bound by |
|--:|--:|--:|--:|--:|--:|--|--|
| 1 | 0.211 ms | 14.1 % | 57.4 % | 5.9 % | 22.5 % | GEMM | small-GEMM fill + launch |
| 2 | 0.078 ms | 23.0 % | 50.5 % | 5.7 % | 20.8 % | GEMM | launch / kernel-body floor |
| 3 | 0.088 ms | 20.9 % | 49.5 % | 5.3 % | 24.3 % | GEMM | launch |
| 4 | 0.112 ms | 17.2 % | 52.0 % | 5.7 % | 25.1 % | GEMM | launch |
| 5 | 0.373 ms | 14.3 % | 53.2 % | 5.6 % | 26.8 % | GEMM | small-GEMM fill |
| 6 | 52.49 ms | 20.3 % | 32.6 % | 8.2 % | 38.9 % | LN+res | **memory bandwidth** |
| 7 | 0.103 ms | 30.6 % | 40.1 % | 6.0 % | 23.3 % | SDPA/GEMM | SDPA + launch (d=32) |
| 8 | 4.327 ms | 6.5 % | 72.5 % | 2.9 % | 18.0 % | GEMM | **compute (165 TFLOP/s)** |
| 9 | 0.210 ms | 13.4 % | 57.9 % | 6.0 % | 22.8 % | GEMM | small-GEMM fill + launch |
| 10 | 0.213 ms | 14.7 % | 56.9 % | 5.9 % | 22.5 % | GEMM | small-GEMM fill + launch |
| 11 | 0.286 ms | 36.5 % | 42.4 % | 4.4 % | 16.7 % | SDPA | attention (16 heads) |
| 12 | 0.123 ms | 24.9 % | 47.3 % | 5.1 % | 22.7 % | GEMM | launch (S=32) |
| 13 | 2.209 ms | 32.6 % | 26.3 % | 6.1 % | 35.0 % | LN+res / SDPA | **SDPA O(S²) + memory** |

Finer GEMM split (EAGER stage timing, rows 1/6/8/13 — cast+QKV vs FFN):
QKV-side ≈ 30 / 29 / 43 / 27 %; FFN-side ≈ 58 / 52 / 53 / 44 % of the eager
forward (`results/g4_9_official_profile_run145.log`).

## 3. Theoretical roofline and what bounds each row

`GEMM floor = 12·M·d²·L / 165.2e12` (165.2 TFLOP/s = fp16 storage / fp32
accumulate = fastest tier that passes `atol=0.002`). `SDPA` taken at its
measured flash time (already accuracy-legal precision). `mem floor =
36·M·d·L / 918 GB/s` (irreducible boundary-crossing traffic; binds only when
`[M,d]` fp32 exceeds ~50 MB of the 72 MB L2). `launch floor = kernels ×
0.855 µs`.

| # | shipped | GEMM-flop floor | mem floor | launch floor | **roofline** | ship/roof | bounded by |
|--:|--:|--:|--:|--:|--:|--:|--|
| 1 | 0.211 ms | 0.039 ms | (L2) | 0.029 ms | **0.069 ms** | 3.1× | compute (sub-roofline small GEMM) |
| 2 | 0.078 ms | 0.001 ms | (L2) | 0.029 ms | **0.029 ms** | 2.7× | launch / kernel-body floor |
| 3 | 0.088 ms | 0.002 ms | (L2) | 0.029 ms | **0.029 ms** | 3.0× | launch |
| 4 | 0.112 ms | 0.003 ms | (L2) | 0.029 ms | **0.029 ms** | 3.8× | launch |
| 5 | 0.373 ms | 0.078 ms | (L2) | 0.029 ms | **0.130 ms** | 2.9× | compute (small GEMM) |
| 6 | 52.49 ms | 6.09 ms | **25.70 ms** | 0.033 ms | **25.70 ms** | 2.0× | **memory bandwidth** |
| 7 | 0.103 ms | 0.002 ms | (L2) | 0.029 ms | **0.033 ms** | 3.1× | SDPA + launch |
| 8 | 4.327 ms | 2.50 ms | (L2) | 0.029 ms | **2.78 ms** | 1.6× | **compute — 165 TFLOP/s roofline** |
| 9 | 0.210 ms | 0.039 ms | (L2) | 0.029 ms | **0.067 ms** | 3.2× | compute (small GEMM) |
| 10 | 0.213 ms | 0.039 ms | (L2) | 0.029 ms | **0.070 ms** | 3.0× | compute (small GEMM) |
| 11 | 0.286 ms | 0.039 ms | (L2) | 0.029 ms | **0.143 ms** | 2.0× | attention (SDPA 37 %) |
| 12 | 0.123 ms | 0.003 ms | (L2) | 0.029 ms | **0.040 ms** | 3.1× | launch (S=32) |
| 13 | 2.209 ms | 0.31 ms | (L2, 34 MB) | 0.029 ms | **1.03 ms** | 2.1× | SDPA O(S²) + LN/residual traffic |

**Reading it.**
- **Rows 2–4, 12** — launch-bound. The GEMMs are 100–1000× below their FLOP
  roofline because an `M×128×128` matmul is 8 K-steps: fill/drain dominates.
  Only lever is fewer/larger kernels → persistent GEMM → needs `wgmma`/TMA
  (Hopper).
- **Rows 1, 5, 9, 10** — same regime, larger M; GEMM is 53–58 % of the forward
  at small-M tensor-core efficiency. ~3× off the analytic roofline; the gap is
  kernel fill + the 22–27 % LN/residual traffic that runs as separate kernels.
- **Rows 7, 11** — SDPA-weighted (d=32; 16 heads). Flash is the right algorithm
  at accuracy-legal precision; SDPA time is its achievable floor.
- **Row 6** — memory-bandwidth-bound. LN+residual+cast (38.9 %) + GELU (8.2 %)
  = 47 % is pure DRAM traffic. Irreducible-traffic model predicts 23.6 GB/fwd;
  the profiler moves 22.7 GB — the elementwise path is at the bandwidth
  roofline. `ship/roof = 2.0×`; the residual is the K=128 thin-GEMM fill
  penalty (persistent GEMM → Hopper).
- **Row 8** — the only cleanly compute-bound shape. GEMM is 72.5 % of the
  forward; cuBLAS runs it at 94–96 % of the 165 TFLOP/s roofline in isolation
  (`docs/PROGRESS.md` §45). `ship/roof = 1.6×`; the in-model excess is L2
  contention between 16 GEMM + flash + 18 elementwise kernels, shown
  unrecoverable without removing the kernel boundaries.
- **Row 13** — SDPA O(S²) (32.6 %) and LN/residual traffic (35.0 %)
  co-dominant. `[M,d]` fits L2 (34 MB) so it is not DRAM-bound; the elementwise
  runs against L2 near its rate.

Full derivation and the "why not faster" argument: `docs/PARETO_FRONTIER_ANALYSIS.md`.
