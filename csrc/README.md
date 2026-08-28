# `csrc/` — hand-written CUDA / C++ / inline-PTX

## Location is load-bearing

`benchmark.py` finds these files relative to its own path —
`_lt_ext()` (~L104) and `_ws_ext()` (~L244) build
`os.path.join(dirname(abspath(__file__)), "csrc", <file>)` and **guard it with
`os.path.exists`**. A missing or moved source returns `None`, the
`torch.ops.g43.*` / `g66.*` ops never register, and the model silently falls
back to eager — the run still passes accuracy, just slower. `csrc/` must stay a
direct child of the repo root, next to `benchmark.py` and the generated
`torch_transformer_benchmark.py` (which carries its own copy of the same lookup).

## What builds at eval time

Only three files. Everything else is investigation scaffolding kept as a receipt.

| File | Stage | Role | Status |
|---|---|---|---|
| `cublaslt_algo.cpp` | G6.6 | cuBLASLt matmul with explicit algorithm selection + fused bias epilogue for the FFN GEMMs (TF32); gated to the tiny non-causal regime | **RUNTIME** |
| `g4_4_warpspec_gemm.cpp` / `.cu` | G4.3 / G4.7 | warp-specialised `mma.sync.m16n8k16` GEMM; the `.cu` also carries the G4.7 fused exact-erf-GELU epilogue. FP32-accumulate arm ships; gated by `G4_7_FFN_CFG` / `_FFN_MIN_TOKENS` and `d_model ≥ 512 ∧ ffn_dim ≥ 2048` (inert on the official matrix) | **RUNTIME** |

## Probe-only (loaded by `experiments/`, never by the eval path)

| File | Stage | Role |
|---|---|---|
| `cublaslt_algo_fp16.cpp` | G6.7 | FP16 variant of the algo search |
| `cublaslt_gelu.cpp` | G4.0 | cuBLASLt matmul + fused GELU/bias epilogue probe |
| `g4_4_mma_gemm.cpp` / `.cu` | G4.4 | tiled `mma.sync` FP16-storage/FP16-accumulate GEMM — verified reference asset |
| `g4_4_mma_micro.cpp` / `.cu` | G4.4 | one-warp 16×16×16 `mma.sync` + `ldmatrix` unit test |
| `l2_persist.cpp` | G2.3 | CUDA L2 cache-persistence control (`set_persist_limit`, `set_window`, …) |

## Closed investigations (kept for the record, not built at eval)

| File | Stage | Outcome |
|---|---|---|
| `g4_5_sass_cfg11.cu` | G4.5 | single-kernel TU isolating G4.4 cfg[11] for SASS round-tripping; toolchain container-format wall, no shippable kernel |
| `g4_6_cutlass_gemm.cpp` / `.cuh`, `g4_6_cutlass_cfg00.cu … cfg23.cu`, `g4_6_cutlass_stock.{cpp,cu}` | G4.6 | CUTLASS bake-off; did not beat the ship gate. **`cfg*.cu` are generated** by `experiments/g4_6_gen_cfgs.py` — edit the generator, not the files |
| `g5_mega_causal.cpp` / `.cu` | G5.MEGA | fused whole-layer causal megakernel, row-6 specialist; ×0.74 (parity, not a win) — Ada shared-memory + occupancy ceiling |

## Header convention

Device kernels carry a top block covering: PROBLEM (shapes/dtypes) · THREAD/WARP
TILING · SHARED-MEMORY LAYOUT (with the Ada 101376 B/CTA cap check) · REGISTER
PRESSURE / OCCUPANCY · PRECISION · STATUS. `g4_4_warpspec_gemm.cu` is the
reference example.
