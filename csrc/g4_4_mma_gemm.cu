// G4.4 -- hand-written tiled GEMM using mma.sync.aligned.m16n8k16.f16.f16
// (FP16 STORAGE, FP16 ACCUMULATE) for the attention path's two projections.
//
// WHY THIS EXISTS. On GeForce Ada (sm_89) FP32-accumulate tensor-core math runs
// at HALF the rate of FP16-accumulate. cuBLAS/cuBLASLt is architecturally
// incapable of ever offering FP16 accumulation (its fp16 path mandates
// CUBLAS_COMPUTE_32F / CUBLAS_COMPUTE_16F is not exposed for these layouts on
// this arch), so this is the ONE mechanism this session has looked at that is
// not competing with cuBLAS inside the same accumulate tier -- every previous
// alternative-kernel attempt (step 19's Triton FFN tile, step 34's cuBLASLt
// algorithm search) lost precisely because it was.
//
//   Out[M,N] = In[M,K] @ W[N,K]^T + bias[N]        (F.linear, all fp16)
//   qkv:      K=512, N=1536
//   out_proj: K=512, N=512
//   M in {64 (tiny), 1024 (default), 8192 (long_seq), 32768 (large_batch)}
//
// NOTE ON THE CEILING: CLAUDE.md's "660 TFLOPS" figure is FP8-storage +
// FP16-accumulate and its own text says never to cite it as available. This
// kernel keeps FP16 STORAGE (no requantisation of anything) and changes only
// the accumulate type, so the honest ceiling is ~2x the FP32-accumulate
// tensor-core rate, i.e. ~330 TFLOPS dense peak, against a measured cuBLASLt
// floor of ~107 TFLOPS (qkv) / ~71 TFLOPS (out_proj) at M=1024.
//
// TILE BUDGET (both operands are 2-byte here -- this is NOT MEGAKERNEL.md's
// FP8 example, recomputed for b_w = b_a = 2):
//   S_per_stage = (BM*BK + BN*BK) * 2 bytes
//   A: BM=64 BN=128 BK=64 -> (4096 + 8192)*2 = 24576 B = 24.0 KB
//        3 stages = 72.0 KB  <= 99 KB OK ;  4 stages = 96.0 KB, +4KB slack
//        overflows the 101376 B limit by 1024 B -> 3 stages
//   B: BM=64 BN=128 BK=32 -> (2048 + 4096)*2 = 12288 B = 12.0 KB
//        7 stages = 84.0 KB <= 99 KB OK ;  8 stages = 96.0 KB also fits but
//        K=512/BK=32 is only 16 iterations so >7 is pointless
//   accumulator: BM*BN in FP16 = 64*128*2 = 16 KB over 256 threads
//        = 16 b32 regs/thread (half what an FP32 accumulator would need).
//        With SPLIT_K the FP32 carry adds 32 more.
//   Actual register counts come from -Xptxas -v, not from this formula.
//
// SPLIT (template param) is MEGAKERNEL.md G4.4's numerics mitigation, halved
// for K=512: accumulate in FP16 within a chunk of SPLIT columns of K, promote
// to an FP32 carry at each chunk boundary. SPLIT=0 disables it (pure FP16 over
// the whole K). This is a WITHIN-BLOCK split, not a grid split -- no partials
// reach HBM and no extra kernel launch.
//
// THREAD / WARP TILING
//   256 threads = 8 warps, laid out 2x4 over the CTA tile (warp tile 32x32).
//   Per k-step each warp issues (32/16)x(32/8) = 2x4 = 8 mma.sync.m16n8k16.
//   K=512 / BK=64 -> 8 outer k-iterations, 4 mma k-steps each.
//
// REGISTER PRESSURE / OCCUPANCY
//   Accumulator: BM*BN fp16 / 256 thr = 16 b32 regs/thread (an FP32 accumulator
//   would be 32; the whole point of the FP16-accumulate tier). +32 for the FP32
//   carry when SPLIT>0. Add ldmatrix fragment regs + the cp.async double-buffer
//   pointers. Shared caps occupancy first: A-tile 3 stages = 72 KB -> 1 CTA/SM
//   (2nd CTA needs 144 KB > 101376 B); B-tile 7 stages = 84 KB -> also 1 CTA/SM.
//   So this kernel runs at 1 block/SM regardless of the register count -- no
//   latency hiding across CTAs, only within the cp.async pipeline. Real
//   -Xptxas -v numbers and the per-config sweep: docs/PROGRESS.md steps 34-37.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define WARP_SIZE 32

namespace {

// ---------------------------------------------------------------------------
// XOR swizzle (MEGAKERNEL.md "Layout details that matter": swizzle, never pad).
//
// A shared tile row is BK halves = BK*2 bytes. The unit of both cp.async and
// ldmatrix is a 16-byte chunk = 8 halves, so CH = BK/8 chunks per row.
// Bank quad of chunk j in row i is ((i*BK*2 + j*16)/4) % 32 = (i*BK/2 + j*4)%32.
// For an ldmatrix 8x8 matrix (8 threads, 8 consecutive rows, one chunk each)
// to be conflict-free those 8 addresses must cover 8 distinct bank quads.
//   BK=64 (CH=8): each row is a full 128 B bank cycle, so the row contributes
//                 nothing to the bank; XOR with (row % 8) spreads the 8 rows
//                 over the 8 chunks.
//   BK=32 (CH=4): a row is 64 B, so rows alternate base-bank 0 / 16 and only
//                 4 chunk positions exist. row%4 does NOT work (rows 0 and 4
//                 collide). The 4 even rows of the group must take 4 distinct
//                 chunks, so the swizzle key must advance once per TWO rows:
//                 XOR with ((row/2) % 4).
// General: key = (row / (8/CH)) % min(CH,8).
// ---------------------------------------------------------------------------
template <int BK>
__device__ __forceinline__ int swz_chunk(int row, int chunk) {
  constexpr int CH = BK / 8;
  constexpr int ROWS_PER_KEY = (CH >= 8) ? 1 : (8 / CH);
  constexpr int KEY_MASK = ((CH < 8) ? CH : 8) - 1;
  return chunk ^ ((row / ROWS_PER_KEY) & KEY_MASK);
}

// offset in HALVES of element (row, chunk*8) inside a [rows][BK] swizzled tile
template <int BK>
__device__ __forceinline__ int swz_off(int row, int chunk) {
  return row * BK + swz_chunk<BK>(row, chunk) * 8;
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

// The FP32-ACCUMULATE twin of the instruction above -- SAME fragment layout,
// SAME operands, SAME everything except the accumulate type. This exists
// purely as the A/B control for the mechanism claim: if this kernel runs at
// the same speed with .f32.f32 as with .f16.f16, then the accumulate tier is
// NOT what binds at this shape and no amount of tuning the FP16-accumulate
// path can recover the 2x the tier is nominally worth.
__device__ __forceinline__ void mma_f32(float (&c)[4], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

}  // namespace

// ---------------------------------------------------------------------------
// The kernel.
//   grid  = (N/BN, M/BM)
//   block = 256 threads = 8 warps, laid out 2 (M) x 4 (N)
//   warp tile = (BM/2) x (BN/4)
// Requires M%BM==0, N%BN==0, K%BK==0 (checked host-side; all real shapes here
// satisfy it -- M in {64,1024,8192,32768}, N in {512,1536}, K=512).
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int NSTAGE, int SPLIT, int ACCF32>
__global__ __launch_bounds__(256) void mma_gemm_kernel(
    const __half *__restrict__ In, const __half *__restrict__ W,
    const __half *__restrict__ bias, __half *__restrict__ Out, int M, int N,
    int K) {
  constexpr int NTHREAD = 256;
  constexpr int NWARP = NTHREAD / WARP_SIZE;   // 8
  constexpr int WARPS_M = 2, WARPS_N = 4;      // 2*4 == NWARP
  constexpr int WM = BM / WARPS_M;             // 32
  constexpr int WN = BN / WARPS_N;             // 32
  constexpr int MT = WM / 16;                  // mma tiles in M per warp
  constexpr int NT = WN / 8;                   // mma tiles in N per warp
  constexpr int CH = BK / 8;                   // 16B chunks per tile row
  constexpr int KSTEP = BK / 16;               // mma k-steps per shared tile
  constexpr int A_CHUNKS = BM * CH;
  constexpr int W_CHUNKS = BN * CH;
  constexpr int A_PER_T = A_CHUNKS / NTHREAD;
  constexpr int W_PER_T = W_CHUNKS / NTHREAD;
  constexpr int A_TILE = BM * BK;              // halves
  constexpr int W_TILE = BN * BK;
  static_assert(WARPS_M * WARPS_N == NWARP, "warp grid");
  static_assert(A_CHUNKS % NTHREAD == 0 && W_CHUNKS % NTHREAD == 0, "loads");
  static_assert(SPLIT == 0 || SPLIT % BK == 0, "SPLIT must be a multiple of BK");

  extern __shared__ __align__(16) char smem_raw[];
  __half *sA = reinterpret_cast<__half *>(smem_raw);
  __half *sW = sA + NSTAGE * A_TILE;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp / WARPS_N;
  const int wn = warp % WARPS_N;

  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;

  // --- global-load index decomposition (constant across the k loop) --------
  int a_row[A_PER_T], a_chunk[A_PER_T], a_dst[A_PER_T];
#pragma unroll
  for (int i = 0; i < A_PER_T; ++i) {
    const int idx = tid + i * NTHREAD;
    a_row[i] = idx / CH;
    a_chunk[i] = idx % CH;
    a_dst[i] = swz_off<BK>(a_row[i], a_chunk[i]);
  }
  int w_row[W_PER_T], w_chunk[W_PER_T], w_dst[W_PER_T];
#pragma unroll
  for (int i = 0; i < W_PER_T; ++i) {
    const int idx = tid + i * NTHREAD;
    w_row[i] = idx / CH;
    w_chunk[i] = idx % CH;
    w_dst[i] = swz_off<BK>(w_row[i], w_chunk[i]);
  }

  auto load_stage = [&](int stage, int kt) {
    const int k0 = kt * BK;
    __half *dA = sA + stage * A_TILE;
    __half *dW = sW + stage * W_TILE;
#pragma unroll
    for (int i = 0; i < A_PER_T; ++i) {
      cp_async_16(smem_u32(dA + a_dst[i]),
                  In + (size_t)(m0 + a_row[i]) * K + k0 + a_chunk[i] * 8);
    }
#pragma unroll
    for (int i = 0; i < W_PER_T; ++i) {
      cp_async_16(smem_u32(dW + w_dst[i]),
                  W + (size_t)(n0 + w_row[i]) * K + k0 + w_chunk[i] * 8);
    }
  };

  // --- accumulators --------------------------------------------------------
  // ACCF32 == 0 : FP16 accumulate (the G4.4 mechanism, 2 b32 regs per tile)
  // ACCF32 == 1 : FP32 accumulate control (4 f32 regs per tile) -- identical
  //               tiling, identical loads, identical everything else.
  constexpr int NH = ACCF32 ? 1 : 2;
  constexpr int NF = ACCF32 ? 4 : 1;
  uint32_t acc[MT][NT][NH];
  float accf[MT][NT][NF];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
      for (int e = 0; e < NH; ++e) acc[i][j][e] = 0;
#pragma unroll
      for (int e = 0; e < NF; ++e) accf[i][j][e] = 0.f;
    }
  // FP32 carry, only materialised when SPLIT != 0 (FP16-accumulate only).
  constexpr int CARRY = (SPLIT != 0 && !ACCF32) ? 1 : 0;
  constexpr int NC = CARRY ? 4 : 1;
  float carry[MT][NT][NC];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < NC; ++e) carry[i][j][e] = 0.f;

  // ldmatrix source addresses for this warp (row within the block tile).
  // A: 16 rows x 16 k-halves per mma tile; lane -> row = lane%16, kchunk half.
  const int a_ld_row_base = wm * WM + (lane & 15);
  const int a_ld_half = (lane >> 4);          // 0 or 1 -> +8 in k
  const int w_ld_row_base = wn * WN + (lane & 7);
  const int w_ld_half = ((lane >> 3) & 1);    // 0 or 1 -> +8 in k

  const int NKT = K / BK;

  // --- prologue: fill NSTAGE-1 stages -------------------------------------
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

#pragma unroll
    for (int ks = 0; ks < KSTEP; ++ks) {
      const int chunk_base = ks * 2;
      uint32_t af[MT][4];
      uint32_t bf[NT][2];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int row = a_ld_row_base + i * 16;
        const int ch = chunk_base + a_ld_half;
        uint32_t addr = smem_u32(tA + swz_off<BK>(row, ch));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3])
            : "r"(addr));
      }
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int row = w_ld_row_base + j * 8;
        const int ch = chunk_base + w_ld_half;
        uint32_t addr = smem_u32(tW + swz_off<BK>(row, ch));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(bf[j][0]), "=r"(bf[j][1])
                     : "r"(addr));
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          if constexpr (ACCF32) {
            mma_f32(reinterpret_cast<float(&)[4]>(accf[i][j]), af[i], bf[j]);
          } else {
            mma_f16(reinterpret_cast<uint32_t(&)[2]>(acc[i][j]), af[i], bf[j],
                    reinterpret_cast<uint32_t(&)[2]>(acc[i][j]));
          }
        }
    }

    // --- split-K numerics mitigation (G4.4) -----------------------------
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

  // --- epilogue ------------------------------------------------------------
  const int gid = lane >> 2;
  const int tig = lane & 3;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int col = n0 + wn * WN + j * 8 + 2 * tig;
      const __half b0 = bias ? bias[col] : __float2half(0.f);
      const __half b1 = bias ? bias[col + 1] : __float2half(0.f);
      __half2 h0, h1;
      if constexpr (ACCF32) {
        // f32 D fragment: {d0,d1} -> (row gid, 2*tig+{0,1});
        //                 {d2,d3} -> (row gid+8, 2*tig+{0,1})
        h0 = __floats2half2_rn(accf[i][j][0], accf[i][j][1]);
        h1 = __floats2half2_rn(accf[i][j][2], accf[i][j][3]);
      } else {
        memcpy(&h0, &acc[i][j][0], 4);
        memcpy(&h1, &acc[i][j][1], 4);
        if constexpr (CARRY) {
          float2 f0 = __half22float2(h0);
          float2 f1 = __half22float2(h1);
          h0 = __floats2half2_rn(f0.x + carry[i][j][0], f0.y + carry[i][j][1]);
          h1 = __floats2half2_rn(f1.x + carry[i][j][2], f1.y + carry[i][j][3]);
        }
      }
      h0 = __hadd2(h0, __halves2half2(b0, b1));
      h1 = __hadd2(h1, __halves2half2(b0, b1));
      const int r0 = m0 + wm * WM + i * 16 + gid;
      *reinterpret_cast<__half2 *>(Out + (size_t)r0 * N + col) = h0;
      *reinterpret_cast<__half2 *>(Out + (size_t)(r0 + 8) * N + col) = h1;
    }
  }
}

// ---------------------------------------------------------------------------
// Host-side dispatch.
//
// G4_4_NO_HOST_DISPATCH lets csrc/g4_5_sass_cfg11.cu include this file to get
// the __global__ template alone (for a single-kernel `nvcc -cubin` build)
// without the switch below force-instantiating all 26 configs. It is a
// preprocessor guard only: when the macro is undefined -- i.e. for every
// existing torch-extension build, including the one step 37 measured -- the
// translation unit is byte-identical to what it was before, so no codegen can
// have changed.
// ---------------------------------------------------------------------------
#ifndef G4_4_NO_HOST_DISPATCH
struct CfgInfo {
  int BM, BN, BK, NSTAGE, SPLIT, ACCF32;
  size_t smem;
};

template <int BM, int BN, int BK, int NSTAGE, int SPLIT, int ACCF32>
static int launch_cfg(const void *In, const void *W, const void *bias,
                      void *Out, int M, int N, int K, cudaStream_t s,
                      size_t *smem_out) {
  const size_t smem = (size_t)NSTAGE * (BM * BK + BN * BK) * sizeof(__half);
  if (smem_out) *smem_out = smem;
  if (M % BM || N % BN || K % BK) return -2;
  if (SPLIT != 0 && (K % (SPLIT ? SPLIT : 1))) return -3;
  auto kern = mma_gemm_kernel<BM, BN, BK, NSTAGE, SPLIT, ACCF32>;
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return -4;
    attr_set = true;
  }
  dim3 grid(N / BN, M / BM);
  kern<<<grid, 256, smem, s>>>(reinterpret_cast<const __half *>(In),
                               reinterpret_cast<const __half *>(W),
                               reinterpret_cast<const __half *>(bias),
                               reinterpret_cast<__half *>(Out), M, N, K);
  return 0;
}

// cfg ids -- keep in sync with kCfg[] in g4_4_mma_gemm.cpp
//   0-9  : the Stage-0 sweep, FP16 accumulate (candidates A and B of the
//          tile-budget arithmetic in this file's header, plus split-K variants)
//  10-14 : bigger BM=128 tile -- doubles the mma:ldmatrix ratio per warp
//          (MT*NT = 4*4 = 16 mma per k-step against 4+4 ldmatrix, vs 2*4 = 8
//          against 2+4 at BM=64), i.e. a fairer shot at the FP16-accumulate
//          ceiling than the BM=64 tiles get
//  15-18 : the FP32-ACCUMULATE CONTROLS. Same tile, same loads, same
//          pipeline; only the mma accumulate type differs. This is the A/B
//          that decides whether the accumulate tier is what binds here.
#define CFG(i, BM, BN, BK, ST, SP, AC) \
  case i: return launch_cfg<BM, BN, BK, ST, SP, AC>(In, W, bias, Out, M, N, K, s, smem_out);
int mma_gemm_launch(int cfg, const void *In, const void *W, const void *bias,
                    void *Out, int M, int N, int K, cudaStream_t s,
                    size_t *smem_out) {
  switch (cfg) {
    CFG(0,  64, 128, 64, 3,   0, 0)
    CFG(1,  64, 128, 64, 2,   0, 0)
    CFG(2,  64, 128, 32, 7,   0, 0)
    CFG(3,  64, 128, 32, 4,   0, 0)
    CFG(4,  64, 128, 32, 5,   0, 0)
    CFG(5,  64, 128, 64, 3, 256, 0)
    CFG(6,  64, 128, 32, 7, 256, 0)
    CFG(7,  64, 128, 64, 3, 128, 0)
    CFG(8,  64, 128, 32, 7, 128, 0)
    CFG(9,  64, 128, 64, 3,  64, 0)
    CFG(10, 128, 128, 64, 3,   0, 0)
    CFG(11, 128, 128, 64, 2,   0, 0)
    CFG(12, 128, 128, 32, 6,   0, 0)
    CFG(13, 128, 128, 32, 4,   0, 0)
    CFG(14, 128, 128, 64, 3, 256, 0)
    CFG(15,  64, 128, 64, 3,   0, 1)
    CFG(16,  64, 128, 64, 2,   0, 1)
    CFG(17, 128, 128, 64, 3,   0, 1)
    CFG(18, 128, 128, 64, 2,   0, 1)
    // Second tuning pass: push arithmetic-per-shared-read further still.
    //   19: warp tile 64x64 -> MT*NT = 4*8 = 32 mma per k-step vs 4+8 ldmatrix
    //   20: warp tile 128x32 -> 8*4 = 32 mma vs 8+4 ldmatrix
    // Both sit at 96 KB of the 99 KB shared budget; see the header arithmetic.
    CFG(19, 128, 256, 64, 2,   0, 0)
    CFG(20, 256, 128, 32, 4,   0, 0)
    CFG(21, 128, 256, 32, 4,   0, 0)
    CFG(22, 256, 128, 64, 2,   0, 0)
    CFG(23, 128, 256, 64, 2,   0, 1)
    CFG(24, 256, 128, 64, 2,   0, 1)
    CFG(25, 128, 256, 64, 2, 256, 0)
    // G8.2: finer FP32 carry. `SPLIT % BK == 0` forces BK=32 to promote more
    // often than every 64 columns of K. At the official matrix's K=128 a
    // SPLIT-64 carry is only 2 chunks; these give 2 and 4, shortening the FP16
    // accumulate chain to 2 and 1 mma k-steps respectively. SPLIT=BK=32 (cfg 27)
    // is the finest carry that is still not FP32 accumulation -- it keeps two
    // k=16 mma steps in FP16 before each promotion.
    CFG(26,  64, 128, 32, 7,  64, 0)
    CFG(27,  64, 128, 32, 7,  32, 0)
    default: return -1;
  }
}
#undef CFG

int mma_gemm_num_cfg() { return 28; }  // G8.2: +cfg26/27 finer carry
#endif  // G4_4_NO_HOST_DISPATCH
