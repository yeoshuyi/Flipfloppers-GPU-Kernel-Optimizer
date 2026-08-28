# Architecture — where things live inside `benchmark.py`

The runtime is one file by design (see `README.md` → "Where is the model
code?"). This maps each conceptual "module" to its location and records the two
guards that keep it in sync with the judges' harness.

Line numbers are approximate anchors into `benchmark.py` at the time of writing;
`grep -n` the symbol name for the exact current line.

## Module map

| Concept | `benchmark.py` | Notes |
|---|---|---|
| **Config** | `class TransformerConfig` (L391) | shape + `causal` flag; `.validate()` |
| **Frozen baseline** | `BaselineSelfAttention` (L420), `BaselineTransformerBlock` (L486), `BaselineTransformer` (L509) | manual matmul→mask→softmax→matmul attention; **DO NOT EDIT** |
| **Frozen harness** | `copy_model_weights` (L1309), `generate_random_case` (L1340), `run_accuracy_tests` (L1465), `benchmark_models` (L1617), `parse_args` (L1700), `main` (L1772) | the scoring loop; **DO NOT EDIT** — `tools/verify_baseline.py` AST-diffs all 20 frozen symbols vs `~/torch_transformer_benchmark.py` |
| **Optimized model** | `class UserOptimizedTransformer(BaselineTransformer)` (L536) … up to `def copy_model_weights` (L1309) | this span is what `tools/sync_entrypoint.py` splices into the judge drop-in (text markers: `class UserOptimizedTransformer` → `def copy_model_weights`) |
| **Custom-op dispatch** | `_lt_ext()` (L104), `_ws_ext()` (L244), `_ffn_register_op()` (L319) | JIT-load the pybind extensions from `csrc/`; register `torch.ops.g43.*` / `g66.*`; each guards its source path with `os.path.exists` → `None` on miss (silent eager fallback) |
| **Env gates** | `_LT_MAX_TOKENS` (L97), `_FFN_CFG` = `G4_7_FFN_CFG` (L239), `_FFN_MIN_TOKENS` = `G4_7_FFN_MIN_TOKENS` (L240) | compile-time constants baked into the traced region |
| **Regime gating** | `forward` (L959) dispatches to `_optimized_forward_causal` (L1073) or `_optimized_forward` (L1187) | `tok = batch·seq` → tiny / default / long-seq / large-batch / padded; the eager `_ensure_*` plan pickers run here, **outside** the compiled region |
| **G1 precompute** | inside `UserOptimizedTransformer.__init__` (within L536–L1073) | LayerNorm-affine fold, QKV fuse, attention-scale fold into `W_Q` (all exact), FP16 weight copies |
| **CUDA-graph capture** | `_compiled_causal` / `_compiled_impl` via `torch.compile(mode="reduce-overhead")` (L1021, L1068) | lazy; per-instance |

## What runs on the official 14-row matrix

Causal path only. Live: G0.1c (SDPA), G0.2c (fused QKV + scale/norm1 fold),
G1.1c (norm2 fold), G6.4bc (FP16 QKV/SDPA/out_proj), G6.4a_v2c (FP16 ffn_in),
G2.4/G2.4b (CUDA graphs). **G4.7c** (the fused-GELU warp-spec kernel) is wired in
but gated `d_model ≥ 512 ∧ ffn_dim ≥ 2048` → inert on every official row
(`ffn_dim ≤ 1024`). So on the scored shapes the causal track calls **zero**
custom kernels — cuBLAS/CUTLASS library GEMMs + PyTorch flash attention +
inductor Triton fusions + graph replay. Details: `docs/DOCUMENTATION.md` §3.3,
`docs/PARETO_FRONTIER_ANALYSIS.md`.

## The two guards (run on every change)

```bash
python3 tools/sync_entrypoint.py          # regenerate torch_transformer_benchmark.py
python3 tools/sync_entrypoint.py --check   # exit 1 if it is stale
python3 tools/verify_baseline.py           # AST-assert the 20 frozen symbols match the judges' script
```

Both resolve the repo root as `dirname(dirname(__file__))` from `tools/` and
read `~/torch_transformer_benchmark.py` (the judges' master copy, outside the
repo). `benchmark.py` and `tools/` must stay at the repo root.
