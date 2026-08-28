// G5.MEGA v3 -- CUTLASS-grade per-sequence fused causal megakernel, row-6
// specialist (B=10000, d_model=128, num_heads=4, head_dim=32, seq_len=128,
// layers=4, ffn_dim=128, causal).
//
// ONE block per sequence.  256 threads = 8 warps.  Threads (2t, 2t+1) share
// token t; each owns HALF the residual row -- xr[64] fp32 in REGISTERS -- for
// the whole 4-layer forward.  x is read from HBM once and written once; the
// residual NEVER round-trips (step 43: that traffic is 38.9% of row 6).
//
// GEMMs, all tensor-core, precision matched to _optimized_forward_causal:
//   qkv / out_proj / ffn_in : mma.sync m16n8k16 f16.f16.f32  (fp16 storage,
//                             fp32 accumulate -- exactly F.linear(fp16)).
//   ffn_out                 : mma.sync m16n8k8  tf32.tf32.f32 (matches the
//                             shipped fp32 nn.Linear under matmul_precision=high).
// attention : online fp32 softmax + max-subtraction, each thread owns 2 heads
//   of its token (>= shipped fp16-flash accuracy; verified g5_5).
// LN : 2-thread cooperative reduction (__shfl_xor).  residual/GELU : thread-local.
//
// 3 x [SEQ][D] fp16 shared buffers (96 KB).

#include <cuda_fp16.h>
#include <math.h>
#include <math_constants.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int D = 128;
constexpr int H = 4;
constexpr int HD = D / H;         // 32
constexpr int SEQ = 128;
constexpr int NL = 4;
constexpr int FF = 128;
constexpr int NT_ = 256;          // threads/block
constexpr int NW = 8;             // warps
constexpr int WROWS = SEQ / NW;   // 16  -> 1 m16 tile per warp
constexpr int KB16 = D / 16;      // 8   (fp16 k-steps)
constexpr int KB8 = D / 8;        // 16  (tf32 k-steps)
constexpr float LN_EPS = 1e-5f;
constexpr float kAlpha = 0.7071067811865475f;

__device__ __forceinline__ float gelu_erf(float v) {
  return v * 0.5f * (1.0f + erff(v * kAlpha));
}

__device__ __forceinline__ void mma16(float (&c)[4], const uint32_t (&a)[4],
                                      const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma8_tf32(float (&c)[4], const uint32_t (&a)[4],
                                          const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- fp16 GEMM: Y[SEQ][N] = Ash(fp16)[SEQ][D] @ Wg(fp16)[N][D]^T (no bias).
// warp w owns rows [w*16 : w*16+16].  One n8-tile at a time.
template <int N, class Sink>
__device__ void gemm16(const __half *Ash, const __half *Wg, int warp, int lane,
                       Sink sink) {
  const int rb = warp * WROWS;
  const int ql = lane >> 2, qk = (lane & 3) * 2;
  constexpr int NTT = N / 8;
#pragma unroll 1
  for (int j = 0; j < NTT; ++j) {
    float acc[4] = {0, 0, 0, 0};
    const int ar = rb + ql;
#pragma unroll
    for (int ks = 0; ks < KB16; ++ks) {
      const int k0 = ks * 16;
      const __half *w = Wg + (size_t)(j * 8 + ql) * D + k0 + qk;
      __half b0[2] = {w[0], w[1]}, b1[2] = {w[8], w[9]};
      uint32_t bf[2] = {*reinterpret_cast<uint32_t *>(b0),
                        *reinterpret_cast<uint32_t *>(b1)};
      const __half *p0 = Ash + ar * D + k0 + qk;
      const __half *p1 = Ash + (ar + 8) * D + k0 + qk;
      __half a0[2] = {p0[0], p0[1]}, a1[2] = {p1[0], p1[1]};
      __half a2[2] = {p0[8], p0[9]}, a3[2] = {p1[8], p1[9]};
      uint32_t af[4] = {*reinterpret_cast<uint32_t *>(a0),
                        *reinterpret_cast<uint32_t *>(a1),
                        *reinterpret_cast<uint32_t *>(a2),
                        *reinterpret_cast<uint32_t *>(a3)};
      mma16(acc, af, bf);
    }
    const int r = rb + ql, c = j * 8 + qk;
    sink(r, c + 0, acc[0]);
    sink(r, c + 1, acc[1]);
    sink(r + 8, c + 0, acc[2]);
    sink(r + 8, c + 1, acc[3]);
  }
}

// ---- fp16 GEMM, BATCHED accumulator (all n-tiles held, epilogue after every
//      A read) -- required when the sink writes back into Ash.  N multiple of 8.
template <int N, class Sink>
__device__ void gemm16b(const __half *Ash, const __half *Wg, int warp, int lane,
                        Sink sink) {
  const int rb = warp * WROWS;
  const int ql = lane >> 2, qk = (lane & 3) * 2;
  constexpr int NTT = N / 8;
  float acc[NTT][4];
#pragma unroll
  for (int j = 0; j < NTT; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
  const int ar = rb + ql;
#pragma unroll
  for (int ks = 0; ks < KB16; ++ks) {
    const int k0 = ks * 16;
    const __half *p0 = Ash + ar * D + k0 + qk;
    const __half *p1 = Ash + (ar + 8) * D + k0 + qk;
    __half a0[2] = {p0[0], p0[1]}, a1[2] = {p1[0], p1[1]};
    __half a2[2] = {p0[8], p0[9]}, a3[2] = {p1[8], p1[9]};
    uint32_t af[4] = {*reinterpret_cast<uint32_t *>(a0),
                      *reinterpret_cast<uint32_t *>(a1),
                      *reinterpret_cast<uint32_t *>(a2),
                      *reinterpret_cast<uint32_t *>(a3)};
#pragma unroll
    for (int j = 0; j < NTT; ++j) {
      const __half *w = Wg + (size_t)(j * 8 + ql) * D + k0 + qk;
      __half b0[2] = {w[0], w[1]}, b1[2] = {w[8], w[9]};
      uint32_t bf[2] = {*reinterpret_cast<uint32_t *>(b0),
                        *reinterpret_cast<uint32_t *>(b1)};
      mma16(acc[j], af, bf);
    }
  }
#pragma unroll
  for (int j = 0; j < NTT; ++j) {
    const int r = rb + ql, c = j * 8 + qk;
    sink(r, c + 0, acc[j][0]);
    sink(r, c + 1, acc[j][1]);
    sink(r + 8, c + 0, acc[j][2]);
    sink(r + 8, c + 1, acc[j][3]);
  }
}

// ---- tf32 GEMM: Y[SEQ][N] = Agelu(fp16 in shared, treated as fp32) [SEQ][D]
//      @ Wf(fp32)[N][D]^T.  GELU is applied to Agelu on read.
template <int N, class Sink>
__device__ void gemm_ffn_out(const __half *Agelu, const float *Wf, int warp,
                             int lane, Sink sink) {
  const int rb = warp * WROWS;
  const int ql = lane >> 2, qk = lane & 3;   // qk in 0..3
  constexpr int NTT = N / 8;
#pragma unroll 1
  for (int j = 0; j < NTT; ++j) {
    float acc[4] = {0, 0, 0, 0};
    const int ar = rb + ql;
#pragma unroll
    for (int ks = 0; ks < KB8; ++ks) {
      const int k0 = ks * 8;
      // B (tf32): b0 = W[j*8+ql][k0+qk], b1 = W[j*8+ql][k0+qk+4]
      float bw0 = Wf[(size_t)(j * 8 + ql) * D + k0 + qk];
      float bw1 = Wf[(size_t)(j * 8 + ql) * D + k0 + qk + 4];
      uint32_t bf[2] = {__float_as_uint(bw0), __float_as_uint(bw1)};
      // A (tf32, gelu'd): a0=A[ar][k0+qk] a1=A[ar+8][k0+qk] a2=A[ar][k0+qk+4] a3=A[ar+8][k0+qk+4]
      float a0 = gelu_erf(__half2float(Agelu[ar * D + k0 + qk]));
      float a1 = gelu_erf(__half2float(Agelu[(ar + 8) * D + k0 + qk]));
      float a2 = gelu_erf(__half2float(Agelu[ar * D + k0 + qk + 4]));
      float a3 = gelu_erf(__half2float(Agelu[(ar + 8) * D + k0 + qk + 4]));
      uint32_t af[4] = {__float_as_uint(a0), __float_as_uint(a1),
                        __float_as_uint(a2), __float_as_uint(a3)};
      mma8_tf32(acc, af, bf);
    }
    const int r = rb + ql, c = j * 8 + (lane & 3) * 2;
    sink(r, c + 0, acc[0]);
    sink(r, c + 1, acc[1]);
    sink(r + 8, c + 0, acc[2]);
    sink(r + 8, c + 1, acc[3]);
  }
}

__global__ __launch_bounds__(256) void mega_kernel(
    const float *__restrict__ xin, float *__restrict__ xout,
    const __half *qkv_w, const __half *qkv_b, const __half *op_w,
    const __half *op_b, const __half *fi_w, const __half *fi_b,
    const float *fo_w, const float *fo_b, const float *fn_w, const float *fn_b,
    int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int tok = tid >> 1;          // 0..127
  const int hf = tid & 1;            // 0 = cols 0..63, 1 = cols 64..127
  const int c0 = hf * 64;            // this thread's residual column base

  extern __shared__ __half smem[];
  __half *sA = smem;
  __half *sB = sA + SEQ * D;
  __half *sC = sB + SEQ * D;

  float xr[64];
  const float *xin_row = xin + (size_t)b * SEQ * D + tok * D + c0;
#pragma unroll
  for (int d = 0; d < 64; ++d) xr[d] = xin_row[d];

  auto ln_to = [&](__half *dst) {
    float ps = 0.f;
#pragma unroll
    for (int d = 0; d < 64; ++d) ps += xr[d];
    ps += __shfl_xor_sync(0xffffffff, ps, 1);          // full row sum
    float mean = ps * (1.f / D);
    float pv = 0.f;
#pragma unroll
    for (int d = 0; d < 64; ++d) { float t = xr[d] - mean; pv += t * t; }
    pv += __shfl_xor_sync(0xffffffff, pv, 1);
    float inv = rsqrtf(pv * (1.f / D) + LN_EPS);
#pragma unroll
    for (int d = 0; d < 64; ++d)
      dst[tok * D + c0 + d] = __float2half((xr[d] - mean) * inv);
  };

  for (int L = 0; L < NL; ++L) {
    const __half *QKVW = qkv_w + (size_t)L * 3 * D * D;
    const __half *QKVB = qkv_b + (size_t)L * 3 * D;
    const __half *OPW = op_w + (size_t)L * D * D;
    const __half *OPB = op_b + (size_t)L * D;
    const __half *FIW = fi_w + (size_t)L * FF * D;
    const __half *FIB = fi_b + (size_t)L * FF;
    const float *FOW = fo_w + (size_t)L * D * FF;
    const float *FOB = fo_b + (size_t)L * D;

    // ---- qkv: n1 -> sA.  q,k via per-tile GEMM -> sB,sC.  v via BATCHED GEMM
    //      -> sA (over n1; batched so all A reads finish before any write). ----
    ln_to(sA);
    __syncthreads();
    gemm16<2 * D>(sA, QKVW, warp, lane, [&](int r, int n, float v) {
      float o = v + __half2float(QKVB[n]);
      if (n < D) sB[r * D + n] = __float2half(o);
      else sC[r * D + (n - D)] = __float2half(o);
    });
    gemm16b<D>(sA, QKVW + (size_t)2 * D * D, warp, lane, [&](int r, int n, float v) {
      sA[r * D + n] = __float2half(v + __half2float(QKVB[2 * D + n]));
    });
    __syncthreads();

    // ---- attention: this thread does heads {2*hf, 2*hf+1} of token `tok` ----
    __half ctx[64];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int h = hf * 2 + hh;
      float m = -CUDART_INF_F, l = 0.f, acc[HD];
#pragma unroll
      for (int e = 0; e < HD; ++e) acc[e] = 0.f;
      const __half *qh = sB + tok * D + h * HD;
      for (int j = 0; j <= tok; ++j) {
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < HD; ++e)
          s += __half2float(qh[e]) * __half2float(sC[j * D + h * HD + e]);
        float mn = fmaxf(m, s);
        float corr = __expf(m - mn), p = __expf(s - mn);
        l = l * corr + p;
#pragma unroll
        for (int e = 0; e < HD; ++e)
          acc[e] = acc[e] * corr + p * __half2float(sA[j * D + h * HD + e]);
        m = mn;
      }
      float linv = 1.f / l;
#pragma unroll
      for (int e = 0; e < HD; ++e) ctx[hh * HD + e] = __float2half(acc[e] * linv);
    }
    __syncthreads();

    // ---- out_proj: ctx -> sB ; GEMM -> sC(fp16) ; add to xr ----
#pragma unroll
    for (int d = 0; d < 64; ++d) sB[tok * D + c0 + d] = ctx[d];
    __syncthreads();
    gemm16<D>(sB, OPW, warp, lane, [&](int r, int n, float v) {
      sC[r * D + n] = __float2half(v + __half2float(OPB[n]));
    });
    __syncthreads();
#pragma unroll
    for (int d = 0; d < 64; ++d) xr[d] += __half2float(sC[tok * D + c0 + d]);

    // ---- FFN ----
    ln_to(sA);                                       // n2 -> sA
    __syncthreads();
    gemm16<FF>(sA, FIW, warp, lane, [&](int r, int n, float v) {
      sB[r * D + n] = __float2half(v + __half2float(FIB[n]));   // ffn_hidden_fp16
    });
    __syncthreads();
    // ffn_out (tf32), GELU(erf) applied inside gemm_ffn_out on the fp16 hidden
    gemm_ffn_out<D>(sB, FOW, warp, lane, [&](int r, int n, float v) {
      sC[r * D + n] = __float2half(v + FOB[n]);
    });
    __syncthreads();
#pragma unroll
    for (int d = 0; d < 64; ++d) xr[d] += __half2float(sC[tok * D + c0 + d]);
    __syncthreads();
  }

  // ---- final_norm (with affine) ----
  float ps = 0.f;
#pragma unroll
  for (int d = 0; d < 64; ++d) ps += xr[d];
  ps += __shfl_xor_sync(0xffffffff, ps, 1);
  float mean = ps * (1.f / D);
  float pv = 0.f;
#pragma unroll
  for (int d = 0; d < 64; ++d) { float t = xr[d] - mean; pv += t * t; }
  pv += __shfl_xor_sync(0xffffffff, pv, 1);
  float inv = rsqrtf(pv * (1.f / D) + LN_EPS);
  float *orow = xout + (size_t)b * SEQ * D + tok * D + c0;
#pragma unroll
  for (int d = 0; d < 64; ++d)
    orow[d] = (xr[d] - mean) * inv * fn_w[c0 + d] + fn_b[c0 + d];
}

}  // namespace

void mega_causal_forward(const torch::Tensor &x, torch::Tensor &out,
                         const torch::Tensor &qkv_w, const torch::Tensor &qkv_b,
                         const torch::Tensor &op_w, const torch::Tensor &op_b,
                         const torch::Tensor &fi_w, const torch::Tensor &fi_b,
                         const torch::Tensor &fo_w, const torch::Tensor &fo_b,
                         const torch::Tensor &fn_w, const torch::Tensor &fn_b) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat, "x f32 cuda");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == SEQ && x.size(2) == D, "x shape");
  const int B = (int)x.size(0);
  size_t smem = (size_t)3 * SEQ * D * sizeof(__half);
  static bool set = false;
  if (!set) {
    cudaFuncSetAttribute(mega_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    set = true;
  }
  mega_kernel<<<B, NT_, smem, c10::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), out.data_ptr<float>(),
      (const __half *)qkv_w.data_ptr(), (const __half *)qkv_b.data_ptr(),
      (const __half *)op_w.data_ptr(), (const __half *)op_b.data_ptr(),
      (const __half *)fi_w.data_ptr(), (const __half *)fi_b.data_ptr(),
      fo_w.data_ptr<float>(), fo_b.data_ptr<float>(),
      fn_w.data_ptr<float>(), fn_b.data_ptr<float>(), B);
  C10_CUDA_CHECK(cudaGetLastError());
}
