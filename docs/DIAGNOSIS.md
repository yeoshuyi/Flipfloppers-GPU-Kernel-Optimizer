# Profiler Fact → Action

Read after profiling. Every proposal must cite a row from this table.

| Observation | Diagnosis | Action |
|---|---|---|
| `gpu_idle_pct` high, `kernel_launches` high | Launch bound | G2.4 → G4.0 → G4.1 |
| `dram_gbps` high, `l2_gbps` low | Weights missing L2 | Arena contiguous (G1.6)? Window on the right stream (G2.3)? |
| `pct_of_peak` < 30% at large shape | TC underutilised | Tile size, G3.7, verify dtype dispatch |
| `stall_long_scoreboard` dominant | Waiting on memory | G3.5; did FP8 free enough shared for +1 stage? |
| `stall_barrier` dominant | Warp imbalance | G4.3 rebalance |
| `bank_conflicts` > 0 | Shared access pattern | G3.4. Do NOT pad. |
| `reg_spills` > 0 | Accumulator too large | Reduce `BM×BN` or raise G3.7 cap |
| Large memcpy kernels in timeline | Layout churn | G0.3 |
| Attention time ∝ S² with HBM traffic | Score matrix materialised | G0.1; confirm `is_causal` not `attn_mask` |
| `occupancy_pct` low, no spills | Shared per block too high | G4.2 K-split |
| Recompiles in `TORCH_LOGS=recompiles` | Shape churn | Mark dynamic dims |

## Accuracy failure → action

| Symptom | Diagnosis | Action |
|---|---|---|
| Fails only at `S≥1024` | Accumulation drift | FP32 softmax accum; check G4.4 split-K chunking |
| Fails uniformly, all shapes | Systematic quantisation | Per-channel scales (G1.5), not per-tensor |
| Fails abs but passes rel | Near-zero elements | Check LN epsilon handling, padded positions |
| `max_abs` moved after an "exact" transform | The fold is wrong | Re-derive G1.x; do not accept |
| Fails only with `--padding-ratio` | Mask path | `masked_fill` elided when it shouldn't be |
