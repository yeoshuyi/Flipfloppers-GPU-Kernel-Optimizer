# Megakernel Reference (G4)

Read only when working on G4. **G1 constant folding must be complete first** —
a megakernel cannot fold affines at runtime, swizzle per iteration, or compute
quantisation scales on the fly.

---

## The shared-memory budget — compute this before proposing any tile

```
S_used = N_stage × (BN·BK·b_w + BM·BK·b_a) + ~4 KB  ≤  101,376 bytes  (99 KB)

accumulator BM×BN FP32 lives in REGISTERS:
    4·BM·BN / (4 · n_threads) regs/thread  ≤  255
```

`b_w` = weight bytes (FP8→1, BF16→2), `b_a` = activation bytes (BF16→2).

### Evaluated at `BM=64, BN=128, BK=64, b_a=2`

| weight dtype | W_tile | A_tile | per stage | 3 stages | 4 stages | 5 stages |
|---|---|---|---|---|---|---|
| BF16 (`b_w=2`) | 16.0 KB | 8.0 KB | 24.0 KB | 72.0 KB OK | **96.0 KB NO** | NO |
| **FP8 (`b_w=1`)** | 8.0 KB | 8.0 KB | 16.0 KB | 48.0 KB OK | **64.0 KB OK** | 80.0 KB OK |

Accumulator both cases: 64×128×4 = 32 KB → **32 regs/thread** at 256 threads.
Comfortable.

> **This is why FP8, and it is not the usual reason.** FP8 is selected for
> **pipeline depth**, not for its 2× arithmetic throughput. At BF16 the identical
> tiling caps at 3 stages (4 overflows by 3 KB). At FP8 the same tiling reaches
> 4 stages with 35 KB spare. Shallow pipelining on Ada costs >30% duty cycle.
> FP8 is the only way to pipeline deeply enough on this chip to realise the
> throughput it also happens to provide.

### Alternative tiles

| BM | BN | BK | b_w | per stage | max stages |
|---|---|---|---|---|---|
| 64 | 128 | 64 | 1 | 16.0 KB | 5 |
| 64 | 128 | 64 | 2 | 24.0 KB | 3 |
| 128 | 128 | 64 | 1 | 24.0 KB | 3 (acc: 64 regs/thread) |
| 64 | 256 | 64 | 1 | 24.0 KB | 3 |
| 32 | 128 | 64 | 1 | 12.0 KB | 7 (poor MMA utilisation) |

Search this space with the agent, but **show the arithmetic** — a tile that does
not fit is not a proposal.

---

## Kernel configuration

```
grid       128 blocks (one per SM), cooperative launch
block      256 threads = 8 warps
warp roles 2 Loader | 4 Consumer | 1 Storer | 1 Controller
pipeline   4-stage cp.async, 16 KB/stage
per layer  FP8 weight slice -> MMA -> grid.sync()
residual   FP32, in registers, across every layer boundary
```

Ada has **no TMA** — async copy is `cp.async` plus a hand-rolled software
pipeline in PTX. Ada has **no `wgmma`** — warp-level `mma.sync.m16n8k16` only.

---

## G4.0 — Two-kernel form. BUILD THIS FIRST.

Same tiling, same pipeline, `grid.sync()` deferred. Not an intermediate tier —
it is the target with one feature removed, and it is a valid stopping point.

1. **Attention block**: LN → fused QKV → flash → `out_proj` → residual
2. **FFN block**: LN → `ffn_in` → GELU in-register → `ffn_out` → residual

12 launches instead of ~60, captured in one CUDA graph ⇒ effectively one launch.

The FFN block is **token-parallel with no cross-token dependency**, so it needs
no grid sync at all. It also keeps the `[1024,2048]` intermediate (4 MB at BF16)
entirely on-chip, eliminating ~48 MB of round-trip traffic across six layers.

**Gate to proceed to G4.1:** `nsys` still shows >15% launch overhead or GPU idle
at the tiny regime *after* CUDA Graphs. If graphs already solved it, stop here.

---

## G4.1 — Persistent kernel

```cpp
void* args[] = {&x, &arena, &out};
cudaLaunchCooperativeKernel((void*)mega, dim3(128), dim3(256),
                            args, smem_bytes, stream);
```

```cpp
__global__ void mega(...) {
    cg::grid_group grid = cg::this_grid();
    float acc[REGS];                      // residual, FP32, never spilled
    for (int layer = 0; layer < 6; ++layer) {
        load_weight_slice(layer);         // cp.async, 4-stage
        attention_block(acc, layer);
        grid.sync();
        ffn_block(acc, layer);
        grid.sync();
    }
}
```

**Preconditions to check in Phase 0:**
- `cudaDevAttrCooperativeLaunch` is non-zero
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` returns ≥1 at your smem/reg
  footprint — a cooperative launch **fails** if the grid cannot be co-resident
- `cudaFuncAttributeMaxDynamicSharedMemorySize` opt-in to 99 KB, explicitly

Work assignment is **static and compile-time**. Do not add a runtime task
scheduler — branch divergence in the dispatch costs more than it saves at this
scale.

---

## G4.2 — K-dimension splitting. Do this before G4.3.

Halve `BK`; each iteration loads only the sub-tile it needs. Peak shared drops
~50%, and the freed pages buy pipeline stages. This is the single highest-value
structural change on Ada, because stages are the binding constraint.

Reported effect on comparable sm_89 work: 64 KB → 32 KB per iteration, page
occupancy 4 → 2, freed capacity spent on depth.

---

## G4.3 — Warp specialisation

| Role | Warps | Job |
|---|---|---|
| Loader | 2 | `cp.async` weight/activation prefetch, N+2 stride |
| Consumer | 4 | `ldmatrix` → `mma.sync` → accumulate |
| Storer | 1 | async writeback, cross-block reduce |
| Controller | 1 | page state, semaphores, dispatch |

Known-good tuning point from published sm_89 results: **Consumer 16→8 warps,
stages 2→4**. Reducing Consumer count shrinks per-GEMM output scale, which
aligns Storer reduce latency with Consumer compute latency and removes the
throughput mismatch that causes stalls.

**Page reuse** (both worth doing):
- *activation → weight*: once activations are in registers, release the page to
  the Loader for weight staging. Deepens the pipeline.
- *activation → output*: MMA accumulates in registers, so the activation page
  can be reallocated for MMA result staging.

---

## G4.4 — FP16 accumulate, the 660 TFLOPS tier

cuBLASLt **cannot** reach this — its FP8 path mandates `CUBLAS_COMPUTE_32F`,
capping at 330. Only hand-written `mma.sync.aligned.m16n8k16.f16.f16` gets there.

**Numerics mitigation is mandatory — split-K:**
```
FP16 accumulate within a K-chunk of 256
FP32 accumulate across chunks
```
Error then grows as √256 = 16 rather than √2048 = 45, a 2.8× reduction in
accumulated rounding. Without this, long-K GEMMs will miss tolerance.

Attempt only after G4.1–G4.3 are stable and passing the full sweep.

---

## G4.5 — Skip the softmax max-subtraction

After G1.2 the scale fold multiplies scores by `d_k^(-1/2)` ≈ 0.125, so scores
are scaled *down*. Overflowing `exp2` in FP32 needs a score above 128, roughly
30σ for a `randn` input distribution.

**Gate it.** Check `input_scale` and shape at dispatch, keep the max-subtracting
path as fallback. Saves one full read pass over the score matrix. Fragile —
this is the first thing to disable if accuracy drifts.

---

## Layout details that matter

**XOR swizzle**, not padding:
```
smem_col = col ^ (row % 8)
```
Conflict-free across 32 banks with no wasted memory. Padding wastes capacity you
do not have and breaks 128-bit alignment, killing `LDG.128`.

**Pre-swizzled weights** (G1.4): permute once at setup into the fragment order
`ldmatrix.sync.aligned.m8n8.x4` expects. Zero runtime permutation, and no need
to write dequantised weights back to shared.

**Small-batch padding**: tensor core instructions need fixed tile dims. At tiny
shapes most of the tile is padding, and loading it wastes HBM→register
bandwidth. Use vector-level loads with explicit register reorganisation rather
than tile-level `ldmatrix` when `BM` exceeds the real token count.

---

## Failure modes, in the order you will hit them

| Symptom | Cause | Fix |
|---|---|---|
| Cooperative launch returns `cudaErrorCooperativeLaunchTooLarge` | Grid not co-resident | Reduce smem or registers until occupancy ≥1 block/SM |
| Hang at `grid.sync()` | Divergent path skips the sync | Every block must reach every sync, unconditionally |
| Correct at B=8, wrong at B=1 | Tile larger than real work | Guard the padding region; check the vector-load path |
| `reg_spills` > 0 | Accumulator too large | Reduce `BM×BN`, or raise the `__launch_bounds__` cap |
| Accuracy drifts only at S≥1024 | FP16 accumulate over long K | G4.4 split-K, or revert to FP32 accumulate |
| Slower than the two-kernel form | Sync overhead exceeds launch savings | The regime does not want a megakernel. Keep G4.0. |

That last row is a legitimate outcome. **G4.0 winning is a result, not a
failure** — record it in the archive and move to another cell.
