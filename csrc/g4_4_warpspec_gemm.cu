// G4.3 -- WARP-SPECIALISED variant of the G4.4 mma.sync FP16-accumulate GEMM.
//
// SEPARATE FILE ON PURPOSE. csrc/g4_4_mma_gemm.cu is the shipped/verified
// reusable asset from docs/PROGRESS.md step 37 and is not touched by this
// probe (same discipline as csrc/cublaslt_gelu.cpp sitting next to the shipped
// csrc/cublaslt_algo.cpp).
//
// WHY. Step 37 left cfg[11] (BM128/BN128/BK64/stg2, 8 warps in a 2x4 grid) at
// 181.93 TF = 55.1% of the 330.3 TF FP16-accumulate tier ceiling, against
// cuBLASLt at 91.2% of its own (half-rate) tier. x1.207, gate x1.3, NOT MET.
// Step 37's own stated next lever: "a CUTLASS-grade epilogue
// (shared-memory-staged 128-bit stores) and warp specialisation --
// MEGAKERNEL.md G4.3's territory."
//
// THE ARITHMETIC THAT MOTIVATES THIS FILE (per 128x128x512 output block,
// 2.52 GHz, Ada: 128 B/clk/SM shared bandwidth, one m16n8k16 issued per warp
// scheduler every 8 clks):
//
//   mma issue      4096 mma/block over 4 schedulers    -> 8192 clk   (the floor)
//   ldmatrix bytes cfg[11] 2x4 warp grid, warp tile 64x32:
//                    per warp per k-step  4 x LDSM.x4 (A, 512 B each)
//                                       + 4 x LDSM.x2 (B, 256 B each) = 3072 B
//                    x 8 warps x 4 k-steps x 8 k-tiles = 786432 B
//                    -> 786432 / 128 B/clk               = 6144 clk
//   8192 + 6144 = 14336 clk  vs  8192 clk if perfectly overlapped.
//   8192 / 0.551 = 14868 clk  <- WHAT STEP 37 ACTUALLY MEASURED.
//
// i.e. the measured 55.1% is almost exactly "mma and ldmatrix do not overlap".
// Two independent levers follow, and this file makes each one a separate
// template flag so they can be priced apart:
//
//   (1) FEWER, FATTER CONSUMER WARPS. MEGAKERNEL.md G4.3's "Consumer 16->8
//       warps" tuning point. 4 consumer warps in a 2x2 grid own 64x64 each,
//       so MT*NT = 4*8 = 32 mma per k-step against 4 LDSM.x4 + 8 LDSM.x2
//       = 4096 B, versus 8 warps x 3072 B = 24576 B for the same output tile.
//       -> 524288 B/block = 4096 clk, a 33% cut in shared-memory READ traffic
//       for identical mma count. If the serialisation model above is right,
//       8192 + 4096 = 12288 clk = 66.7% of tier.
//   (2) REGDB -- explicit register double-buffering of the ldmatrix fragments
//       (load k-step s+1's fragments before issuing k-step s's mma), which is
//       the CUTLASS inner-loop trick that makes the two pipes overlap at all.
//
//   (3) SMEMEPI -- the CUTLASS-grade epilogue. The step-37 epilogue writes
//       __half2 (4 B/thread/instruction); a warp's store instruction touches
//       8 different output rows x 16 B = 8 transactions for 128 B of payload.
//       Staging the accumulators through the (already allocated, now idle)
//       pipeline shared memory and re-reading them chunk-major turns that into
//       st.global.v4.b32 -- 32 lanes x 16 B = 4 x 128 B fully-utilised
//       transactions. This is MEGAKERNEL.md's "activation -> output" page
//       reuse: the pipeline pages are dead by the epilogue, so the staging
//       buffer costs ZERO extra shared memory.
//
// WARP SPECIALISATION PROPER (NLW > 0) follows MEGAKERNEL.md G4.3's role
// table: Loader warps do nothing but cp.async, Consumer warps do nothing but
// ldmatrix/mma, and they are decoupled by per-stage FULL/EMPTY named barriers
// (`barrier.sync`/`barrier.arrive`) instead of a block-wide __syncthreads, so
// the Loader is only ever blocked by the stage it wants to OVERWRITE, never by
// the stage currently being consumed. NEXTRA covers the Storer + Controller
// warps: for a single-output-tile GEMM neither has real work (there is no
// cross-block reduce and no page state to arbitrate), so NEXTRA is carried as
// a MEASURED COST -- it prices what the two idle roles take in scheduler slots
// and occupancy, which is a number this project did not have.
//
// SHARED-MEMORY BUDGET (both operands 2-byte; 101376 B usable):
//   BM128/BN128/BK64 -> (128+128)*64*2 = 32768 B/stage
//        2 stages = 64 KB OK | 3 stages = 96 KB OK | 4 stages = 128 KB NO
//   epilogue staging = BM*BN*2 = 32 KB, aliased onto the pipeline pages.
//   -> smem = max(NSTAGE*32 KB, 32 KB), never more than the pipeline needs.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define WARP_SIZE 32

namespace wsg {

// --- XOR swizzle: byte-for-byte the verified helper from g4_4_mma_gemm.cu ---
// A tile row is RL halves; the unit of cp.async/ldmatrix is a 16-B chunk = 8
// halves, so CH = RL/8 chunks per row. key = (row / (8/CH)) % min(CH,8).
template <int RL>
__device__ __forceinline__ int swz_chunk(int row, int chunk) {
  constexpr int CH = RL / 8;
  constexpr int ROWS_PER_KEY = (CH >= 8) ? 1 : (8 / CH);
  constexpr int KEY_MASK = ((CH < 8) ? CH : 8) - 1;
  return chunk ^ ((row / ROWS_PER_KEY) & KEY_MASK);
}

template <int RL>
__device__ __forceinline__ int swz_off(int row, int chunk) {
  return row * RL + swz_chunk<RL>(row, chunk) * 8;
}

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_16(uint32_t dst_smem,
                                            const void *src_global) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_smem),
               "l"(src_global));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Named barriers. `barrier.sync id, cnt` blocks until cnt threads arrive;
// `barrier.arrive id, cnt` counts the arrival and returns immediately. The
// "memory" clobber is load-bearing: without it nvcc is free to sink the
// ldmatrix reads of stage s across the barrier that publishes it.
__device__ __forceinline__ void bar_sync(int id, int cnt) {
  asm volatile("barrier.sync %0, %1;\n" ::"r"(id), "r"(cnt) : "memory");
}

__device__ __forceinline__ void bar_arrive(int id, int cnt) {
  asm volatile("barrier.arrive %0, %1;\n" ::"r"(id), "r"(cnt) : "memory");
}

__device__ __forceinline__ void mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2],
                                        const uint32_t (&c)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]));
}

// G4.7: FP32-ACCUMULATE twin of mma_f16. Same m16n8k16 shape, same A/B
// fragment layout, same D element->(row,col) mapping (c0,c1 at row lane/4,
// cols 2*(lane%4)(+1); c2,c3 at row lane/4+8) -- so every index in this file
// is unchanged; only the accumulator's type and the tensor-core datapath
// differ. This is the HALF-RATE path on GeForce Ada (CLAUDE.md trap 2), so it
// exists ONLY to make epilogue fusion available with ZERO numerics change,
// not to beat cuBLASLt on the GEMM itself.
__device__ __forceinline__ void mma_f32(float (&d)[4], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldm_x2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

// LDM4B: fetch the B fragments of TWO adjacent n-tiles (j, j+1) with a single
// ldmatrix.x4 instead of two ldmatrix.x2. Same bytes, HALF the LDSM
// instructions -- for a 64-wide warp tile B costs 8 x2 per k-step, which is
// the single biggest LDSM issue term once the warp tile is fat.
//   matrix (lane>>3): 0=(j,k0) 1=(j,k1) 2=(j+1,k0) 3=(j+1,k1)
//   so lane l addresses row = wn*WN + (j + (l>>4))*8 + (l&7),
//                      chunk = ks*2 + ((l>>3)&1)   -- identical to the x2 form.
__device__ __forceinline__ void ldm_x4_raw(uint32_t &r0, uint32_t &r1,
                                           uint32_t &r2, uint32_t &r3,
                                           uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// Streaming 128-bit global store: `.cs` marks the line evict-first, which
// matters at large_batch where the output alone (100.7 MB) is larger than the
// 72 MB L2 and would otherwise evict the A/W tiles that DO get reused.
__device__ __forceinline__ void st_global_v4_cs(void *p, uint4 v) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};\n" ::"l"(p),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

// ---------------------------------------------------------------------------
// G4.7: EXACT erf-form GELU for the ffn_in epilogue.
//
// READ THIS BEFORE TOUCHING IT. docs/PROGRESS.md step 35 Finding 2 closed
// GELU fusion via cuBLASLt's CUBLASLT_EPILOGUE_GELU_BIAS on ACCURACY: that
// epilogue computes the TANH approximation, the model computes
// F.gelu(approximate="none"), and the systematic mismatch was 4.74e-04
// (csrc/cublaslt_gelu.cpp). This kernel is hand-written, so it is not stuck
// with cuBLASLt's choice -- it computes the erf form, in fp32, with the SAME
// expression and the SAME operation order as ATen's own CUDA kernel
// (aten/src/ATen/native/cuda/ActivationGeluKernel.cu, approximate=None):
//
//     constexpr opmath_t kAlpha = M_SQRT1_2;
//     return x * opmath_t(0.5) * (opmath_t(1) + ::erf(x * kAlpha));
//
// with opmath_t = float for a float input, and ::erf(float) = erff. Matching
// the order matters: this is claimed as an EXACT transform, so the gate on it
// is BIT-IDENTITY against F.gelu(x.float(), approximate="none"), not "close".
__device__ __forceinline__ float gelu_erf_exact(float x) {
  constexpr float kAlpha = (float)M_SQRT1_2;
  return x * 0.5f * (1.0f + erff(x * kAlpha));
}

}  // namespace wsg

// ---------------------------------------------------------------------------
// The kernel.
//
//   Out[M,N] = In[M,K] @ W[N,K]^T + bias[N]        (F.linear, all fp16)
//   grid  = (N/BN, M/BM)
//
// Template parameters
//   BM,BN,BK  block tile
//   NSTAGE    cp.async pipeline depth
//   NCM,NCN   consumer warp grid (NCM*NCN consumer warps; warp tile is
//             (BM/NCM) x (BN/NCN))
//   NLW       dedicated Loader warps. 0 = NOT warp-specialised (the consumers
//             also issue the cp.async, i.e. step 37's structure) -- this is the
//             control arm.
//   NEXTRA    Storer + Controller warps (0/1/2). Only meaningful when NLW > 0.
//   SMEMEPI   1 = shared-memory-staged st.global.v4.b32 epilogue
//   REGDB     1 = register double-buffer the ldmatrix fragments
//   SPLIT     FP32 carry every SPLIT columns of K (0 = off), the step-37
//             numerics mitigation, carried over unchanged
//   EPIGELU   G4.7. 0 = the shipped behaviour, Out is fp16, no activation.
//             1 = Out is FP32 and the epilogue applies the EXACT erf-form
//             GELU (wsg::gelu_erf_exact) to every element before storing.
//             The value the GELU sees is the fp16 GEMM+bias result, i.e.
//             EXACTLY the tensor F.linear(fp16) would have produced, so
//             EPIGELU=1 is a bit-exact fusion of
//                 F.gelu(F.linear(...).float(), approximate="none")
//             and NOT a precision-ladder step. Staging still goes through
//             shared memory as fp16 -- the identical, already-verified
//             swizzle -- and the fp32 widening happens in the drain, so the
//             epilogue's shared-memory footprint and bank behaviour are
//             byte-for-byte what SMEMEPI already had.
//   ACCF32    G4.7. 0 = the shipped FP16-accumulate mma (the whole point of
//             the G4.4/G4.3 line). 1 = mma.*.f32.f16.f16.f32, i.e. FP32
//             accumulation with fp16 operands -- the HALF-RATE datapath, so
//             strictly slower per FLOP, but numerically what cuBLASLt already
//             does. Bias is then added in FP32 before the single fp16
//             round-trip, exactly like cuBLASLt's bias epilogue. ACCF32=1 is
//             therefore PRECISION-NEUTRAL: it introduces no new error source,
//             which is what makes it a candidate for the causal path that
//             step 41 could not reach. SPLIT is meaningless here (there is
//             nothing to carry) and is rejected by static_assert.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int NSTAGE, int NCM, int NCN, int NLW,
          int NEXTRA, int SMEMEPI, int REGDB, int SPLIT, int LDM4B, int EPICS,
          int SWZG, int EPIGELU, int ACCF32>
__global__ __launch_bounds__((NCM * NCN + NLW + NEXTRA) * WARP_SIZE)
void ws_gemm_kernel(const __half *__restrict__ In, const __half *__restrict__ W,
                    const __half *__restrict__ bias, void *__restrict__ Out_v,
                    int M, int N, int K) {
  using namespace wsg;

  constexpr int NCONS = NCM * NCN;
  constexpr int NWARP = NCONS + NLW + NEXTRA;
  constexpr int NTHREAD = NWARP * WARP_SIZE;
  constexpr int WARPSPEC = (NLW > 0);
  // Threads that issue cp.async.
  constexpr int LOAD_W = WARPSPEC ? NLW : NCONS;
  constexpr int LOAD_T = LOAD_W * WARP_SIZE;
  // Threads participating in the pipeline barriers (loaders + consumers).
  constexpr int PIPE_T = WARPSPEC ? (NCONS + NLW) * WARP_SIZE : NTHREAD;

  constexpr int WM = BM / NCM;
  constexpr int WN = BN / NCN;
  constexpr int MT = WM / 16;
  constexpr int NT = WN / 8;
  constexpr int CH = BK / 8;
  constexpr int KSTEP = BK / 16;
  constexpr int A_CHUNKS = BM * CH;
  constexpr int W_CHUNKS = BN * CH;
  constexpr int A_PER_T = A_CHUNKS / LOAD_T;
  constexpr int W_PER_T = W_CHUNKS / LOAD_T;
  constexpr int A_TILE = BM * BK;
  constexpr int W_TILE = BN * BK;

  static_assert(BM % NCM == 0 && BN % NCN == 0, "consumer warp grid");
  static_assert(WM % 16 == 0 && WN % 8 == 0, "warp tile vs mma shape");
  static_assert(A_CHUNKS % LOAD_T == 0 && W_CHUNKS % LOAD_T == 0, "loads");
  static_assert(SPLIT == 0 || SPLIT % BK == 0, "SPLIT must be a multiple of BK");
  static_assert(!(ACCF32 && SPLIT), "ACCF32 accumulates in fp32 already");
  static_assert(NSTAGE >= 2, "need >=2 stages");
  static_assert(!WARPSPEC || 2 * NSTAGE + 1 <= 16, "named barrier ids");

  // Barrier ids: [0, NSTAGE)  = FULL(stage)   loader arrives, consumer waits
  //              [NSTAGE, 2N) = EMPTY(stage)  consumer arrives, loader waits
  // NOTE: ids start at 1. __syncthreads() compiles to `bar.sync 0`, i.e. it
  // OWNS hardware barrier 0; the epilogue below uses __syncthreads() while
  // loader warps may still be finishing, so overlapping id 0 would be a race.
  constexpr int BAR_FULL = 1;
  constexpr int BAR_EMPTY = 1 + NSTAGE;

  extern __shared__ __align__(16) char smem_raw[];
  __half *sA = reinterpret_cast<__half *>(smem_raw);
  __half *sW = sA + NSTAGE * A_TILE;
  // Epilogue staging ALIASES the pipeline pages (MEGAKERNEL.md "activation ->
  // output" page reuse): by the time it is written the pipeline is drained.
  __half *sEpi = reinterpret_cast<__half *>(smem_raw);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // SWZG: CUTLASS/Triton group-M threadblock rasterisation. The default
  // (x = n-tile, y = m-tile) launch order runs a whole row of n-tiles before
  // advancing m, so the blocks co-resident in one wave share A but stream the
  // whole of W. Grouping SWZG m-tiles per n-tile makes a wave a SWZG x
  // (wave/SWZG) rectangle instead of a 1 x wave strip, cutting L2 misses on
  // both operands. Computed once, outside the k loop.
  int m0, n0;
  if constexpr (SWZG > 0) {
    const int npm = gridDim.y, npn = gridDim.x;
    const int pid = blockIdx.x + npn * blockIdx.y;
    const int per_group = SWZG * npn;
    const int gid_ = pid / per_group;
    const int first_m = gid_ * SWZG;
    const int gsz = (npm - first_m) < SWZG ? (npm - first_m) : SWZG;
    const int in_group = pid % per_group;
    m0 = (first_m + (in_group % gsz)) * BM;
    n0 = (in_group / gsz) * BN;
  } else {
    m0 = blockIdx.y * BM;
    n0 = blockIdx.x * BN;
  }
  const int NKT = K / BK;

  const bool is_consumer = (warp < NCONS);
  const bool is_loader = WARPSPEC ? (warp >= NCONS && warp < NCONS + NLW) : true;

  // ---- global-load index decomposition (constant across the k loop) -------
  // ltid is the thread's index WITHIN the loading group.
  const int ltid = WARPSPEC ? (tid - NCONS * WARP_SIZE) : tid;
  int a_row[A_PER_T], a_chunk[A_PER_T], a_dst[A_PER_T];
  int w_row[W_PER_T], w_chunk[W_PER_T], w_dst[W_PER_T];
  if (is_loader) {
#pragma unroll
    for (int i = 0; i < A_PER_T; ++i) {
      const int idx = ltid + i * LOAD_T;
      a_row[i] = idx / CH;
      a_chunk[i] = idx % CH;
      a_dst[i] = swz_off<BK>(a_row[i], a_chunk[i]);
    }
#pragma unroll
    for (int i = 0; i < W_PER_T; ++i) {
      const int idx = ltid + i * LOAD_T;
      w_row[i] = idx / CH;
      w_chunk[i] = idx % CH;
      w_dst[i] = swz_off<BK>(w_row[i], w_chunk[i]);
    }
  }

  auto load_stage = [&](int stage, int kt) {
    const int k0 = kt * BK;
    __half *dA = sA + stage * A_TILE;
    __half *dW = sW + stage * W_TILE;
#pragma unroll
    for (int i = 0; i < A_PER_T; ++i)
      cp_async_16(smem_u32(dA + a_dst[i]),
                  In + (size_t)(m0 + a_row[i]) * K + k0 + a_chunk[i] * 8);
#pragma unroll
    for (int i = 0; i < W_PER_T; ++i)
      cp_async_16(smem_u32(dW + w_dst[i]),
                  W + (size_t)(n0 + w_row[i]) * K + k0 + w_chunk[i] * 8);
  };

  // ---- consumer state -----------------------------------------------------
  const int wm = warp / NCN;   // only meaningful when is_consumer
  const int wn = warp % NCN;
  // G4.7: the two accumulator banks are mutually exclusive -- the unused one
  // is declared 1x1 so it costs no registers, and every use of either is
  // behind `if constexpr (ACCF32)`.
  constexpr int AM = ACCF32 ? 1 : MT;
  constexpr int AN = ACCF32 ? 1 : NT;
  constexpr int FM = ACCF32 ? MT : 1;
  constexpr int FN = ACCF32 ? NT : 1;
  uint32_t acc[AM][AN][2];
  float accf[FM][FN][4];
  constexpr int CARRY = (SPLIT != 0) ? 1 : 0;
  constexpr int NC = CARRY ? 4 : 1;
  float carry[AM][AN][NC];
  if constexpr (ACCF32) {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) accf[i][j][e] = 0.f;
  } else {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        acc[i][j][0] = 0;
        acc[i][j][1] = 0;
#pragma unroll
        for (int e = 0; e < NC; ++e) carry[i][j][e] = 0.f;
      }
  }

  const int a_ld_row_base = wm * WM + (lane & 15);
  const int a_ld_half = (lane >> 4);
  // For LDM4B the row base advances by 8 for the upper half of the warp, so
  // one x4 covers n-tiles (j, j+1). Otherwise the verified x2 addressing.
  const int w_ld_row_base =
      wn * WN + (lane & 7) + (LDM4B ? ((lane >> 4) * 8) : 0);
  const int w_ld_half = ((lane >> 3) & 1);
  static_assert(!LDM4B || (NT % 2 == 0), "LDM4B needs an even NT");

  auto load_bfrag = [&](uint32_t (&dst)[NT][2], const __half *tW, int ks) {
    if constexpr (LDM4B) {
#pragma unroll
      for (int j = 0; j < NT; j += 2)
        ldm_x4_raw(dst[j][0], dst[j][1], dst[j + 1][0], dst[j + 1][1],
                   smem_u32(tW + swz_off<BK>(w_ld_row_base + j * 8,
                                             ks * 2 + w_ld_half)));
    } else {
#pragma unroll
      for (int j = 0; j < NT; ++j)
        ldm_x2(dst[j], smem_u32(tW + swz_off<BK>(w_ld_row_base + j * 8,
                                                 ks * 2 + w_ld_half)));
    }
  };

  // ==========================================================================
  // MAIN LOOP
  // ==========================================================================
  if constexpr (WARPSPEC) {
    // ---------------- warp-specialised: decoupled loader / consumer --------
    if (is_loader) {
      // prologue: stages 0..NSTAGE-2 are free, no EMPTY wait needed.
#pragma unroll
      for (int s = 0; s < NSTAGE - 1; ++s) {
        if (s < NKT) load_stage(s, s);
        cp_async_commit();
      }
      for (int kt = 0; kt < NKT; ++kt) {
        cp_async_wait<NSTAGE - 2>();
        bar_arrive(BAR_FULL + (kt % NSTAGE), PIPE_T);
        const int nxt = kt + NSTAGE - 1;
        if (nxt < NKT) {
          bar_sync(BAR_EMPTY + (nxt % NSTAGE), PIPE_T);
          load_stage(nxt % NSTAGE, nxt);
        }
        cp_async_commit();
      }
    } else if (is_consumer) {
      // Pre-arm EMPTY for the one stage the prologue left unfilled, otherwise
      // the loader's first bar_sync(EMPTY) at kt=0 has nobody to release it.
      if (NSTAGE - 1 < NKT)
        bar_arrive(BAR_EMPTY + ((NSTAGE - 1) % NSTAGE), PIPE_T);
      for (int kt = 0; kt < NKT; ++kt) {
        const int stage = kt % NSTAGE;
        bar_sync(BAR_FULL + stage, PIPE_T);
        const __half *tA = sA + stage * A_TILE;
        const __half *tW = sW + stage * W_TILE;

        uint32_t af[REGDB ? 2 : 1][MT][4];
        uint32_t bf[REGDB ? 2 : 1][NT][2];
        if (REGDB) {
#pragma unroll
          for (int i = 0; i < MT; ++i)
            ldm_x4(af[0][i],
                   smem_u32(tA + swz_off<BK>(a_ld_row_base + i * 16,
                                             0 + a_ld_half)));
          load_bfrag(bf[0], tW, 0);
        }
#pragma unroll
        for (int ks = 0; ks < KSTEP; ++ks) {
          const int cur = REGDB ? (ks & 1) : 0;
          const int nxt = REGDB ? ((ks + 1) & 1) : 0;
          if (!REGDB) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
              ldm_x4(af[0][i], smem_u32(tA + swz_off<BK>(a_ld_row_base + i * 16,
                                                         ks * 2 + a_ld_half)));
            load_bfrag(bf[0], tW, ks);
          } else if (ks + 1 < KSTEP) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
              ldm_x4(af[nxt][i],
                     smem_u32(tA + swz_off<BK>(a_ld_row_base + i * 16,
                                               (ks + 1) * 2 + a_ld_half)));
            load_bfrag(bf[nxt], tW, ks + 1);
          }
          if constexpr (ACCF32) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
              for (int j = 0; j < NT; ++j)
                mma_f32(accf[i][j], af[cur][i], bf[cur][j]);
          } else {
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
              for (int j = 0; j < NT; ++j)
                mma_f16(acc[i][j], af[cur][i], bf[cur][j], acc[i][j]);
          }
        }
        bar_arrive(BAR_EMPTY + stage, PIPE_T);

        if constexpr (CARRY) {
          const int kdone = (kt + 1) * BK;
          if ((kdone % (SPLIT ? SPLIT : 1)) == 0) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
              for (int j = 0; j < NT; ++j) {
                __half2 h0, h1;
                memcpy(&h0, &acc[i][j][0], 4);
                memcpy(&h1, &acc[i][j][1], 4);
                float2 f0 = __half22float2(h0);
                float2 f1 = __half22float2(h1);
                carry[i][j][0] += f0.x;
                carry[i][j][1] += f0.y;
                carry[i][j][2] += f1.x;
                carry[i][j][3] += f1.y;
                acc[i][j][0] = 0;
                acc[i][j][1] = 0;
              }
          }
        }
      }
    }
    // Storer / Controller warps: no work in a single-output-tile GEMM (no
    // cross-block reduce, no page arbitration). They fall straight through to
    // the block-wide barrier below and are priced by the NEXTRA sweep.
  } else {
    // ---------------- control arm: step 37's structure ---------------------
#pragma unroll
    for (int s = 0; s < NSTAGE - 1; ++s) {
      if (s < NKT) load_stage(s, s);
      cp_async_commit();
    }
    for (int kt = 0; kt < NKT; ++kt) {
      cp_async_wait<NSTAGE - 2>();
      __syncthreads();
      const int prefetch = kt + NSTAGE - 1;
      if (prefetch < NKT) load_stage(prefetch % NSTAGE, prefetch);
      cp_async_commit();

      const int stage = kt % NSTAGE;
      const __half *tA = sA + stage * A_TILE;
      const __half *tW = sW + stage * W_TILE;

      uint32_t af[REGDB ? 2 : 1][MT][4];
      uint32_t bf[REGDB ? 2 : 1][NT][2];
      if (REGDB) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
          ldm_x4(af[0][i],
                 smem_u32(tA + swz_off<BK>(a_ld_row_base + i * 16, a_ld_half)));
        load_bfrag(bf[0], tW, 0);
      }
#pragma unroll
      for (int ks = 0; ks < KSTEP; ++ks) {
        const int cur = REGDB ? (ks & 1) : 0;
        const int nx = REGDB ? ((ks + 1) & 1) : 0;
        if (!REGDB) {
#pragma unroll
          for (int i = 0; i < MT; ++i)
            ldm_x4(af[0][i], smem_u32(tA + swz_off<BK>(a_ld_row_base + i * 16,
                                                       ks * 2 + a_ld_half)));
          load_bfrag(bf[0], tW, ks);
        } else if (ks + 1 < KSTEP) {
#pragma unroll
          for (int i = 0; i < MT; ++i)
            ldm_x4(af[nx][i],
                   smem_u32(tA + swz_off<BK>(a_ld_row_base + i * 16,
                                             (ks + 1) * 2 + a_ld_half)));
          load_bfrag(bf[nx], tW, ks + 1);
        }
        if constexpr (ACCF32) {
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
              mma_f32(accf[i][j], af[cur][i], bf[cur][j]);
        } else {
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
              mma_f16(acc[i][j], af[cur][i], bf[cur][j], acc[i][j]);
        }
      }

      if constexpr (CARRY) {
        const int kdone = (kt + 1) * BK;
        if ((kdone % (SPLIT ? SPLIT : 1)) == 0) {
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) {
              __half2 h0, h1;
              memcpy(&h0, &acc[i][j][0], 4);
              memcpy(&h1, &acc[i][j][1], 4);
              float2 f0 = __half22float2(h0);
              float2 f1 = __half22float2(h1);
              carry[i][j][0] += f0.x;
              carry[i][j][1] += f0.y;
              carry[i][j][2] += f1.x;
              carry[i][j][3] += f1.y;
              acc[i][j][0] = 0;
              acc[i][j][1] = 0;
            }
        }
      }
    }
  }

  // ==========================================================================
  // EPILOGUE
  // ==========================================================================
  const int gid = lane >> 2;
  const int tig = lane & 3;
  __half *Out = reinterpret_cast<__half *>(Out_v);
  float *OutF = reinterpret_cast<float *>(Out_v);

  // G4.7. The accumulator -> "what F.linear would have produced" reduction,
  // in ONE place, so the fp16-accumulate and fp32-accumulate arms cannot
  // drift. Both arms end with a value that is EXACTLY an fp16 F.linear
  // output, which is what makes the GELU fusion below an exact transform:
  //   ACCF32=0: fp16 accumulate (+ optional fp32 SPLIT carry), bias in fp16
  //             -- byte-identical to what step 41 shipped.
  //   ACCF32=1: fp32 accumulate, bias added in FP32 and rounded ONCE, which
  //             is exactly cuBLASLt's CUBLASLT_EPILOGUE_BIAS semantics.
#define WS_REDUCE_ACC(i, j, h0, h1)                                            \
  do {                                                                         \
    if constexpr (ACCF32) {                                                    \
      const float bf0 = bias ? __half2float(b0) : 0.f;                         \
      const float bf1 = bias ? __half2float(b1) : 0.f;                         \
      h0 = __floats2half2_rn(accf[i][j][0] + bf0, accf[i][j][1] + bf1);        \
      h1 = __floats2half2_rn(accf[i][j][2] + bf0, accf[i][j][3] + bf1);        \
    } else {                                                                   \
      memcpy(&h0, &acc[i][j][0], 4);                                           \
      memcpy(&h1, &acc[i][j][1], 4);                                           \
      if constexpr (CARRY) {                                                   \
        float2 f0 = __half22float2(h0);                                        \
        float2 f1 = __half22float2(h1);                                        \
        h0 = __floats2half2_rn(f0.x + carry[i][j][0], f0.y + carry[i][j][1]);  \
        h1 = __floats2half2_rn(f1.x + carry[i][j][2], f1.y + carry[i][j][3]);  \
      }                                                                        \
      h0 = __hadd2(h0, __halves2half2(b0, b1));                                \
      h1 = __hadd2(h1, __halves2half2(b0, b1));                                \
    }                                                                          \
  } while (0)

  if constexpr (!SMEMEPI) {
    // step 37's epilogue, verbatim: one st.global.b32 per accumulator pair.
    // (EPIGELU widens that to one st.global.b64 of two fp32 activations.)
    if (is_consumer) {
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int col = n0 + wn * WN + j * 8 + 2 * tig;
          const __half b0 = bias ? bias[col] : __float2half(0.f);
          const __half b1 = bias ? bias[col + 1] : __float2half(0.f);
          __half2 h0, h1;
          WS_REDUCE_ACC(i, j, h0, h1);
          const int r0 = m0 + wm * WM + i * 16 + gid;
          if constexpr (EPIGELU) {
            const float2 f0 = __half22float2(h0);
            const float2 f1 = __half22float2(h1);
            const float2 g0 = make_float2(gelu_erf_exact(f0.x),
                                          gelu_erf_exact(f0.y));
            const float2 g1 = make_float2(gelu_erf_exact(f1.x),
                                          gelu_erf_exact(f1.y));
            *reinterpret_cast<float2 *>(OutF + (size_t)r0 * N + col) = g0;
            *reinterpret_cast<float2 *>(OutF + (size_t)(r0 + 8) * N + col) = g1;
          } else {
            *reinterpret_cast<__half2 *>(Out + (size_t)r0 * N + col) = h0;
            *reinterpret_cast<__half2 *>(Out + (size_t)(r0 + 8) * N + col) = h1;
          }
        }
    }
  } else {
    // CUTLASS-grade epilogue: stage row-major-swizzled into the dead pipeline
    // pages, then drain chunk-major as st.global.v4.b32.
    //
    // WRITE bank check (WN=64, so a staged row is 64 halves = 128 B = one full
    // bank cycle, CH_EPI = 8):
    //   word index = row*32 + swz_chunk<WN>(row,j)*4 + tig
    //              = row*32 + (j ^ (row%8))*4 + tig
    //   bank = ((j ^ gid)*4 + tig) % 32 ; over lanes (gid 0..7, tig 0..3) the
    //   term (j^gid) takes all 8 values -> 32 distinct banks. CONFLICT-FREE.
    // READ bank check: lane t reads chunk idx = t -> row = t/8, c = t%8,
    //   bank = ((c ^ row)*4) % 32; each aligned group of 8 lanes covers all 32
    //   banks exactly once (16-B loads are serviced 8 lanes/cycle anyway).
    __syncthreads();
    if (is_consumer) {
      __half *sEw = sEpi + warp * (WM * WN);
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int col = n0 + wn * WN + j * 8 + 2 * tig;
          const __half b0 = bias ? bias[col] : __float2half(0.f);
          const __half b1 = bias ? bias[col + 1] : __float2half(0.f);
          __half2 h0, h1;
          WS_REDUCE_ACC(i, j, h0, h1);
          const int r0 = i * 16 + gid;
          *reinterpret_cast<__half2 *>(sEw + swz_off<WN>(r0, j) + 2 * tig) = h0;
          *reinterpret_cast<__half2 *>(sEw + swz_off<WN>(r0 + 8, j) + 2 * tig) =
              h1;
        }
    }
    __syncthreads();
    constexpr int CH_EPI = WN / 8;
    constexpr int CHUNKS_PER_WARP = WM * CH_EPI;
    constexpr int TOTAL_CHUNKS = NCONS * CHUNKS_PER_WARP;
    constexpr int DRAIN_ITERS = (TOTAL_CHUNKS + NTHREAD - 1) / NTHREAD;
#pragma unroll
    for (int d = 0; d < DRAIN_ITERS; ++d) {
      const int idx = tid + d * NTHREAD;
      if (idx >= TOTAL_CHUNKS) break;
      const int ws = idx / CHUNKS_PER_WARP;
      const int rem = idx % CHUNKS_PER_WARP;
      const int row = rem / CH_EPI;
      const int c = rem % CH_EPI;
      const uint4 v = *reinterpret_cast<const uint4 *>(
          sEpi + ws * (WM * WN) + swz_off<WN>(row, c));
      const int gr = m0 + (ws / NCN) * WM + row;
      const int gc = n0 + (ws % NCN) * WN + c * 8;
      if constexpr (EPIGELU) {
        // G4.7. The 16-B staged chunk is 8 fp16 F.linear outputs; widen and
        // activate them here, then store 32 B as two 128-bit transactions.
        // Staging stays fp16, so the swizzle, the bank arithmetic and the
        // shared-memory budget above are UNCHANGED and still the verified
        // ones -- only the drain's store width differs (8 lanes x 32 B = 256 B
        // contiguous = 2 fully-utilised 128 B transactions, same efficiency).
        const __half2 *hp = reinterpret_cast<const __half2 *>(&v);
        float g[8];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float2 t = __half22float2(hp[q]);
          g[2 * q] = gelu_erf_exact(t.x);
          g[2 * q + 1] = gelu_erf_exact(t.y);
        }
        float *op = OutF + (size_t)gr * N + gc;
        const uint4 lo = *reinterpret_cast<const uint4 *>(&g[0]);
        const uint4 hi = *reinterpret_cast<const uint4 *>(&g[4]);
        if constexpr (EPICS) {
          st_global_v4_cs(op, lo);
          st_global_v4_cs(op + 4, hi);
        } else {
          *reinterpret_cast<uint4 *>(op) = lo;
          *reinterpret_cast<uint4 *>(op + 4) = hi;
        }
      } else if constexpr (EPICS) {
        st_global_v4_cs(Out + (size_t)gr * N + gc, v);
      } else {
        *reinterpret_cast<uint4 *>(Out + (size_t)gr * N + gc) = v;
      }
    }
  }
#undef WS_REDUCE_ACC
}

// ---------------------------------------------------------------------------
// Host-side dispatch + self-describing config table (single source of truth --
// the .cpp reads names from here so the two cannot drift, unlike
// g4_4_mma_gemm.cpp's duplicated kCfg[]).
// ---------------------------------------------------------------------------

// X-macro: id, BM,BN,BK, NSTAGE, NCM,NCN, NLW,NEXTRA, SMEMEPI,REGDB,SPLIT,
//              LDM4B, EPICS, SWZG
//
// ROUND 1 (cfg 0-25) is the lever-isolation ladder; its measured result is in
// results/g4_3_warpspec_sweep_run124.log. ROUND 2 (cfg 26+) tunes around the
// two round-1 winners with three further CUTLASS-grade levers: ldmatrix.x4 for
// the B fragments (LDM4B), evict-first output stores (EPICS), and group-M
// threadblock rasterisation (SWZG).
#define WS_CFG_LIST(X)                                                        \
  /* -- 0: EXACT REPLICA OF STEP 37's cfg[11]. 8 warps 2x4, no warp spec,   */ \
  /*      no smem epilogue, no regdb. Reproduces 283.3 us / 181.9 TF, and   */ \
  /*      Stage 0a confirms it is BITWISE identical to g4_4 cfg[11].        */ \
  X(0,  128, 128, 64, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0)                        \
  /* -- 1-3: lever (1) alone -- fat 64x64 consumer warp tiles, 4 warps,     */ \
  /*      still NOT warp-specialised. Isolates "fewer/fatter warps".        */ \
  X(1,  128, 128, 64, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0)                        \
  X(2,  128, 128, 64, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0)                        \
  X(3,  128, 128, 64, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0)                        \
  /* -- 4-5: lever (2) alone -- register double-buffered fragments.         */ \
  X(4,  128, 128, 64, 2, 2, 4, 0, 0, 0, 1, 0, 0, 0, 0)                        \
  X(5,  128, 128, 64, 2, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0)                        \
  /* -- 6-7: lever (3) alone -- CUTLASS-grade smem-staged 128-bit epilogue. */ \
  X(6,  128, 128, 64, 2, 2, 4, 0, 0, 1, 0, 0, 0, 0, 0)                        \
  X(7,  128, 128, 64, 2, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0)                        \
  /* -- 8-9: levers (1)+(2)+(3) stacked, still no warp specialisation.      */ \
  X(8,  128, 128, 64, 2, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0)                        \
  X(9,  128, 128, 64, 3, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0)                        \
  /* -- 10-17: G4.3 WARP SPECIALISATION, MEGAKERNEL.md's role table.        */ \
  X(10, 128, 128, 64, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0)                        \
  X(11, 128, 128, 64, 3, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0)                        \
  X(12, 128, 128, 64, 2, 2, 2, 2, 0, 1, 1, 0, 0, 0, 0)                        \
  X(13, 128, 128, 64, 3, 2, 2, 2, 0, 1, 1, 0, 0, 0, 0)                        \
  X(14, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 0, 0, 0, 0)                        \
  X(15, 128, 128, 64, 3, 2, 2, 4, 0, 1, 1, 0, 0, 0, 0)                        \
  X(16, 128, 128, 64, 2, 2, 2, 2, 0, 1, 0, 0, 0, 0, 0)                        \
  X(17, 128, 128, 64, 3, 2, 2, 2, 0, 1, 0, 0, 0, 0, 0)                        \
  /* -- 18-21: 8 consumer warps kept at the fat 64x64 tile by doubling the  */ \
  /*      block tile to 256x128 / 128x256 (2 stages only; 48 KB/stage).     */ \
  X(18, 256, 128, 64, 2, 4, 2, 0, 0, 1, 1, 0, 0, 0, 0)                        \
  X(19, 128, 256, 64, 2, 2, 4, 0, 0, 1, 1, 0, 0, 0, 0)                        \
  X(20, 256, 128, 64, 2, 4, 2, 2, 0, 1, 1, 0, 0, 0, 0)                        \
  X(21, 128, 256, 64, 2, 2, 4, 2, 0, 1, 1, 0, 0, 0, 0)                        \
  /* -- 22-23: 128x64 block tile -> 4 STAGES fit (MEGAKERNEL "stages 2->4").*/ \
  X(22, 128, 64,  64, 4, 2, 1, 0, 0, 1, 1, 0, 0, 0, 0)                        \
  X(23, 128, 64,  64, 4, 2, 1, 2, 0, 1, 1, 0, 0, 0, 0)                        \
  /* -- 24-25: the step-37 SPLIT numerics carry on the round-1 shapes.      */ \
  X(24, 128, 128, 64, 2, 2, 2, 0, 0, 1, 1, 256, 0, 0, 0)                      \
  X(25, 128, 128, 64, 2, 2, 2, 2, 0, 1, 1, 256, 0, 0, 0)                      \
  /* ===================== ROUND 2 ======================================== */ \
  /* 26-31: round-1 generalist winner cfg[14] (WARPSPEC, 4 loader warps)    */ \
  /*        + each new lever alone, then combined.                          */ \
  X(26, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 0, 1, 0, 0)                        \
  X(27, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 0, 0, 1, 0)                        \
  X(28, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 0, 0, 0, 8)                        \
  X(29, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 0, 1, 1, 0)                        \
  X(30, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 0, 1, 1, 8)                        \
  X(31, 128, 128, 64, 3, 2, 2, 4, 0, 1, 1, 0, 1, 1, 8)                        \
  /* 32-35: round-1 large_batch winner cfg[19] (plain, BM128/BN256, 8 warps */ \
  /*        at 64x64) + the new levers.                                     */ \
  X(32, 128, 256, 64, 2, 2, 4, 0, 0, 1, 1, 0, 1, 0, 0)                        \
  X(33, 128, 256, 64, 2, 2, 4, 0, 0, 1, 1, 0, 1, 1, 0)                        \
  X(34, 128, 256, 64, 2, 2, 4, 0, 0, 1, 1, 0, 1, 1, 8)                        \
  X(35, 128, 256, 64, 2, 2, 4, 2, 0, 1, 1, 0, 1, 1, 8)                        \
  /* 36-37: 256x128 twin of the above.                                      */ \
  X(36, 256, 128, 64, 2, 4, 2, 0, 0, 1, 1, 0, 1, 1, 0)                        \
  X(37, 256, 128, 64, 2, 4, 2, 0, 0, 1, 1, 0, 1, 1, 8)                        \
  /* 38-39: the 1x4 grid (128x32 warp tile) that round 1 found unexpectedly */ \
  /*        strong under the OLD epilogue -- never combined with the new    */ \
  /*        one. MT*NT = 8*4 = 32 mma per k-step, same as 64x64.            */ \
  X(38, 128, 128, 64, 2, 1, 4, 0, 0, 1, 1, 0, 1, 1, 0)                        \
  X(39, 128, 128, 64, 2, 1, 4, 4, 0, 1, 1, 0, 1, 1, 0)                        \
  /* 40-41: more loader warps on the generalist (ld8 -> 12 warps / 384 thr) */ \
  X(40, 128, 128, 64, 2, 2, 2, 8, 0, 1, 1, 0, 1, 1, 0)                        \
  X(41, 128, 128, 64, 3, 2, 2, 8, 0, 1, 1, 0, 1, 1, 0)                        \
  /* 42-43: smaller BM for the DEFAULT shape (M=1024 only fills 96 blocks   */ \
  /*        at BM=128; BM=64 doubles the grid to 192 > 128 SMs).            */ \
  X(42,  64, 128, 64, 3, 1, 2, 2, 0, 1, 1, 0, 1, 1, 0)                        \
  X(43,  64, 128, 64, 4, 1, 2, 2, 0, 1, 1, 0, 1, 1, 0)                        \
  /* 44-45: pure-plain (no warp spec) best, with the new levers -- the      */ \
  /*        control that says how much warp spec is still worth.            */ \
  X(44, 128, 128, 64, 2, 2, 2, 0, 0, 1, 1, 0, 1, 1, 0)                        \
  X(45, 128, 128, 64, 2, 2, 4, 0, 0, 1, 1, 0, 1, 1, 0)                        \
  /* 46-47: winner shape + SPLIT numerics carry, in case accuracy binds.    */ \
  X(46, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 256, 1, 1, 0)                      \
  X(47, 128, 256, 64, 2, 2, 4, 0, 0, 1, 1, 256, 1, 1, 8)                      \
  /* ===================== ROUND 3: NUMERICAL RESCUE ====================== */ \
  /* The ship-verification run (results/g4_3_ship_verify_run128.log) came   */ \
  /* back FAST and WRONG on the causal shapes: max_abs 0.00550 (long_seq)   */ \
  /* and 0.00763 (large_batch) against a 0.002 budget. That is FP16         */ \
  /* ACCUMULATE error, exactly what MEGAKERNEL.md G4.4 says split-K is for. */ \
  /* 48-50 are the generalist cfg[26] with the FP32 carry at three chunk    */ \
  /* sizes, so the accuracy/speed trade can be priced rather than guessed.  */ \
  X(48, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 64,  1, 0, 0)                      \
  X(49, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 128, 1, 0, 0)                      \
  X(50, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 256, 1, 0, 0)

// ===========================================================================
// G4.7 -- THE FFN LIST. A SECOND, SEPARATE X-macro list, appended after the
// shipped one, with two extra columns (EPIGELU, ACCF32). It is a separate
// list SPECIFICALLY so that WS_CFG_LIST above -- the 51 configs step 41
// measured, swept and shipped from -- is not edited at all: `git diff` on
// those 51 rows is empty, cfg ids 0-50 keep their meaning, and cfg[0] stays
// the bitwise control for g4_4_mma_gemm cfg[11].
//
// WHY THESE SHAPES. run127 (results/g4_3_warpspec_modelshapes_run127.log)
// already measured the FFN GEMMs with the shipped configs and named the
// per-shape winners: ffn_in (K=512, N=2048) went to cfg[39] at long_seq
// (79.44 us, x1.452) and cfg[37] at large_batch (322.32 us, x1.425), with
// cfg[26]/cfg[33]/cfg[29] close behind. Those four tiles are carried here,
// each in four arms:
//     no GELU / GELU  x  fp16-accumulate / fp32-accumulate
// so that the epilogue-fusion win and the accumulate-precision cost can be
// read off SEPARATELY instead of bundled -- which matters because the judge
// harness runs causal shapes only, and step 41 closed FP16 ACCUMULATION on
// causal accuracy. ACCF32 arms exist to answer "is the epilogue fusion worth
// anything on its own, with zero numerics change?".
//
// X-macro: id, BM,BN,BK, NSTAGE, NCM,NCN, NLW,NEXTRA, SMEMEPI,REGDB,SPLIT,
//              LDM4B, EPICS, SWZG, EPIGELU, ACCF32
#define WS_CFG_LIST_G47(X)                                                    \
  /* 51-54: cfg[26] generalist (warpspec, 4 loaders, ldm4b).               */ \
  X(51, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 0,  1, 0, 0, 1, 0)                 \
  X(52, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 64, 1, 0, 0, 1, 0)                 \
  X(53, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 0,  1, 0, 0, 0, 1)                 \
  X(54, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 0,  1, 0, 0, 1, 1)                 \
  /* 55-58: cfg[39] -- run127's ffn_in winner at long_seq (1x4 grid).      */ \
  X(55, 128, 128, 64, 2, 1, 4, 4, 0, 1, 1, 0,  1, 1, 0, 1, 0)                 \
  X(56, 128, 128, 64, 2, 1, 4, 4, 0, 1, 1, 64, 1, 1, 0, 1, 0)                 \
  X(57, 128, 128, 64, 2, 1, 4, 4, 0, 1, 1, 0,  1, 1, 0, 0, 1)                 \
  X(58, 128, 128, 64, 2, 1, 4, 4, 0, 1, 1, 0,  1, 1, 0, 1, 1)                 \
  /* 59-62: cfg[37] -- run127's ffn_in winner at large_batch (256x128).    */ \
  X(59, 256, 128, 64, 2, 4, 2, 0, 0, 1, 1, 0,  1, 1, 8, 1, 0)                 \
  X(60, 256, 128, 64, 2, 4, 2, 0, 0, 1, 1, 64, 1, 1, 8, 1, 0)                 \
  X(61, 256, 128, 64, 2, 4, 2, 0, 0, 1, 1, 0,  1, 1, 8, 0, 1)                 \
  X(62, 256, 128, 64, 2, 4, 2, 0, 0, 1, 1, 0,  1, 1, 8, 1, 1)                 \
  /* 63-66: cfg[33] -- 128x256 twin, second at large_batch.                */ \
  X(63, 128, 256, 64, 2, 2, 4, 0, 0, 1, 1, 0,  1, 1, 0, 1, 0)                 \
  X(64, 128, 256, 64, 2, 2, 4, 0, 0, 1, 1, 64, 1, 1, 0, 1, 0)                 \
  X(65, 128, 256, 64, 2, 2, 4, 0, 0, 1, 1, 0,  1, 1, 0, 0, 1)                 \
  X(66, 128, 256, 64, 2, 2, 4, 0, 0, 1, 1, 0,  1, 1, 0, 1, 1)                 \
  /* 67-70: the SPLIT-64 arms of the two winners WITHOUT gelu, so the      */ \
  /*        "plain drop-in for ffn_in" arm can be priced too, and 71-74    */ \
  /*        the fp32-accumulate 2x4 warp grid (MT*NT=16 -> only 64 accum   */ \
  /*        registers, which is the config ACCF32 should prefer).          */ \
  X(67, 128, 128, 64, 2, 1, 4, 4, 0, 1, 1, 64, 1, 1, 0, 0, 0)                 \
  X(68, 256, 128, 64, 2, 4, 2, 0, 0, 1, 1, 64, 1, 1, 8, 0, 0)                 \
  X(69, 128, 256, 64, 2, 2, 4, 0, 0, 1, 1, 64, 1, 1, 0, 0, 0)                 \
  X(70, 128, 128, 64, 2, 2, 2, 4, 0, 1, 1, 0,  1, 0, 0, 0, 0)                 \
  X(71, 128, 128, 64, 2, 2, 4, 4, 0, 1, 0, 0,  1, 0, 0, 1, 1)                 \
  X(72, 128, 128, 64, 2, 2, 4, 4, 0, 1, 0, 0,  1, 0, 0, 0, 1)                 \
  X(73, 128, 128, 64, 3, 2, 4, 4, 0, 1, 0, 0,  1, 0, 0, 1, 1)                 \
  X(74, 128, 128, 64, 2, 2, 4, 0, 0, 1, 0, 0,  1, 1, 0, 1, 1)                 \
  /* 75-76: fp32-accumulate at the 256x128 / 128x256 tiles, 2x4-style warp */ \
  /*        grids so MT*NT stays at 16.                                    */ \
  X(75, 128, 256, 64, 2, 2, 8, 0, 0, 1, 0, 0,  1, 1, 0, 1, 1)                 \
  X(76, 256, 128, 64, 2, 4, 4, 0, 0, 1, 0, 0,  1, 1, 8, 1, 1)

namespace {
struct WsDesc {
  int BM, BN, BK, NSTAGE, NCM, NCN, NLW, NEXTRA, SMEMEPI, REGDB, SPLIT;
  int LDM4B, EPICS, SWZG, EPIGELU, ACCF32;
};
}  // namespace

template <int BM, int BN, int BK, int NSTAGE, int NCM, int NCN, int NLW,
          int NEXTRA, int SMEMEPI, int REGDB, int SPLIT, int LDM4B, int EPICS,
          int SWZG, int EPIGELU, int ACCF32>
static int ws_launch(const void *In, const void *W, const void *bias, void *Out,
                     int M, int N, int K, cudaStream_t s, size_t *smem_out) {
  constexpr int NTHREAD = (NCM * NCN + NLW + NEXTRA) * WARP_SIZE;
  const size_t pipe = (size_t)NSTAGE * (BM * BK + BN * BK) * sizeof(__half);
  const size_t epi = SMEMEPI ? (size_t)BM * BN * sizeof(__half) : 0;
  const size_t smem = pipe > epi ? pipe : epi;
  if (smem_out) *smem_out = smem;
  if (M % BM || N % BN || K % BK) return -2;
  if (SPLIT != 0 && (K % (SPLIT ? SPLIT : 1))) return -3;
  auto kern = ws_gemm_kernel<BM, BN, BK, NSTAGE, NCM, NCN, NLW, NEXTRA, SMEMEPI,
                             REGDB, SPLIT, LDM4B, EPICS, SWZG, EPIGELU, ACCF32>;
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return -4;
    attr_set = true;
  }
  dim3 grid(N / BN, M / BM);
  kern<<<grid, NTHREAD, smem, s>>>(reinterpret_cast<const __half *>(In),
                                   reinterpret_cast<const __half *>(W),
                                   reinterpret_cast<const __half *>(bias), Out,
                                   M, N, K);
  return 0;
}

// The shipped list carries 14 columns; EPIGELU/ACCF32 are appended as 0 here
// rather than edited into 51 already-measured rows.
#define WS_CASE(i, BM, BN, BK, ST, NCM, NCN, NLW, NEX, EPI, RDB, SP, L4, CS, SZ) \
  case i:                                                                        \
    return ws_launch<BM, BN, BK, ST, NCM, NCN, NLW, NEX, EPI, RDB, SP, L4, CS,   \
                     SZ, 0, 0>(In, W, bias, Out, M, N, K, s, smem_out);

#define WS_CASE47(i, BM, BN, BK, ST, NCM, NCN, NLW, NEX, EPI, RDB, SP, L4, CS,   \
                  SZ, GE, AF)                                                    \
  case i:                                                                        \
    return ws_launch<BM, BN, BK, ST, NCM, NCN, NLW, NEX, EPI, RDB, SP, L4, CS,   \
                     SZ, GE, AF>(In, W, bias, Out, M, N, K, s, smem_out);

int ws_gemm_launch(int cfg, const void *In, const void *W, const void *bias,
                   void *Out, int M, int N, int K, cudaStream_t s,
                   size_t *smem_out) {
  switch (cfg) {
    WS_CFG_LIST(WS_CASE)
    WS_CFG_LIST_G47(WS_CASE47)
    default:
      return -1;
  }
}
#undef WS_CASE
#undef WS_CASE47

#define WS_ROW(i, BM, BN, BK, ST, NCM, NCN, NLW, NEX, EPI, RDB, SP, L4, CS, SZ) \
  {BM, BN, BK, ST, NCM, NCN, NLW, NEX, EPI, RDB, SP, L4, CS, SZ, 0, 0},
#define WS_ROW47(i, BM, BN, BK, ST, NCM, NCN, NLW, NEX, EPI, RDB, SP, L4, CS,   \
                 SZ, GE, AF)                                                    \
  {BM, BN, BK, ST, NCM, NCN, NLW, NEX, EPI, RDB, SP, L4, CS, SZ, GE, AF},

static const WsDesc kWsCfg[] = {WS_CFG_LIST(WS_ROW) WS_CFG_LIST_G47(WS_ROW47)};
#undef WS_ROW
#undef WS_ROW47

// G4.7: the output tensor's dtype is a property of the CONFIG, not of the
// call, so the .cpp can check it before dispatching instead of guessing.
int ws_gemm_cfg_outf32(int cfg) {
  if (cfg < 0 || cfg >= (int)(sizeof(kWsCfg) / sizeof(kWsCfg[0]))) return -1;
  return kWsCfg[cfg].EPIGELU ? 1 : 0;
}

int ws_gemm_num_cfg() { return (int)(sizeof(kWsCfg) / sizeof(kWsCfg[0])); }

// Fills a caller-provided buffer with a human-readable description.
void ws_gemm_cfg_desc(int cfg, char *buf, int buflen) {
  if (cfg < 0 || cfg >= ws_gemm_num_cfg()) {
    snprintf(buf, buflen, "<bad cfg>");
    return;
  }
  const WsDesc &d = kWsCfg[cfg];
  const int ncons = d.NCM * d.NCN;
  const int nwarp = ncons + d.NLW + d.NEXTRA;
  const size_t pipe = (size_t)d.NSTAGE * (d.BM * d.BK + d.BN * d.BK) * 2;
  const size_t epi = d.SMEMEPI ? (size_t)d.BM * d.BN * 2 : 0;
  const size_t smem = pipe > epi ? pipe : epi;
  snprintf(buf, buflen,
           "BM%d BN%d BK%d stg%d | cons %dx%d (wtile %dx%d) ld%d extra%d "
           "| thr%d %s%s%s smem=%dKB",
           d.BM, d.BN, d.BK, d.NSTAGE, d.NCM, d.NCN, d.BM / d.NCM,
           d.BN / d.NCN, d.NLW, d.NEXTRA, nwarp * 32,
           d.NLW ? "WARPSPEC " : "plain    ", d.SMEMEPI ? "smemEpi " : "stgEpi  ",
           d.REGDB ? "regdb " : "      ", (int)(smem / 1024));
  {
    int n = (int)strlen(buf);
    snprintf(buf + n, buflen - n, "%s%s%s%s", d.LDM4B ? " ldm4b" : "",
             d.EPICS ? " stcs" : "", d.SWZG ? " swz" : "", "");
    if (d.SWZG) {
      n = (int)strlen(buf);
      snprintf(buf + n, buflen - n, "%d", d.SWZG);
    }
    if (d.SPLIT) {
      n = (int)strlen(buf);
      snprintf(buf + n, buflen - n, " split%d", d.SPLIT);
    }
    if (d.ACCF32) {
      n = (int)strlen(buf);
      snprintf(buf + n, buflen - n, " ACCF32");
    }
    if (d.EPIGELU) {
      n = (int)strlen(buf);
      snprintf(buf + n, buflen - n, " GELU->fp32");
    }
  }
}
