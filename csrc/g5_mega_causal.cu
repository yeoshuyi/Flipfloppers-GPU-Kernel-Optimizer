// G5.MEGA v2 -- per-sequence fused causal transformer megakernel, row-6
// specialist (B=10000, d_model=128, num_heads=4, seq_len=128, layers=4).
//
// ONE block per sequence; 128 threads = 4 warps.  The fp32 residual row `x[D]`
// lives in REGISTERS (thread t owns query row t) for the whole 4-layer
// forward -- zero HBM traffic for the residual (step 43: 38.9% of row 6).
// Shared = two [SEQ][D] fp16 buffers sK, sV (64 KB).  qkv output round-trips a
// per-block [SEQ][3D] fp16 global scratch (the d_model=128 shared budget won't
// hold n1 + q + k + v at once).
//
// qkv / out_proj / ffn_in : mma.sync m16n8k16 f32-accumulate (matches the
//   shipped fp16-storage/fp32-accum GEMMs).  A from shared, B streamed from L2.
// ffn_out : scalar fp32, GELU(erf) inline on the fp16-rounded hidden --
//   matches F.gelu(ffn_hidden_fp16.to(fp32)) then the fp32 nn.Linear.
// attention : scalar per query row, online fp32 softmax + max-subtraction.
// Precision matched op-for-op to benchmark.py::_optimized_forward_causal
// (attention runs fp32 rather than fp16 flash -- >= shipped accuracy,
//  verified in probes/g5_5_mega_correct.py).

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
constexpr int NW = 4;
constexpr int WROWS = SEQ / NW;   // 32
constexpr int MT = WROWS / 16;    // 2
constexpr int KB = D / 16;        // 8
constexpr float LN_EPS = 1e-5f;
constexpr float kAlpha = 0.7071067811865475f;

__device__ __forceinline__ float gelu_erf(float v) {
  return v * 0.5f * (1.0f + erff(v * kAlpha));
}

__device__ __forceinline__ void mma_m16n8k16(float (&c)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Y[SEQ][N] = Ash[SEQ][D](fp16) @ Wg[N][D](fp16, row-major)^T   (no bias).
// Warp `warp` owns rows [warp*WROWS : +WROWS].  One n8-tile at a time so the
// live accumulator is MT*4 floats.  Ash MUST stay valid for the whole call
// (the sink must not write back into Ash).  N multiple of 8.
template <int N, class Sink>
__device__ void gemm_rowreg(const __half *Ash, const __half *Wg, int warp,
                            int lane, Sink sink) {
  const int rb = warp * WROWS;
  const int qlo = lane >> 2;
  const int qk = (lane & 3) * 2;
  constexpr int NT = N / 8;
#pragma unroll 1
  for (int j = 0; j < NT; ++j) {
    float acc[MT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][e] = 0.f;
#pragma unroll
    for (int ks = 0; ks < KB; ++ks) {
      const int k0 = ks * 16;
      const __half *w = Wg + (size_t)(j * 8 + qlo) * D + k0 + qk;
      __half b0[2] = {w[0], w[1]}, b1[2] = {w[8], w[9]};
      uint32_t bf[2] = {*reinterpret_cast<uint32_t *>(b0),
                        *reinterpret_cast<uint32_t *>(b1)};
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int ar = rb + i * 16 + qlo;
        const __half *p0 = Ash + ar * D + k0 + qk;
        const __half *p1 = Ash + (ar + 8) * D + k0 + qk;
        __half a0[2] = {p0[0], p0[1]}, a1[2] = {p1[0], p1[1]};
        __half a2[2] = {p0[8], p0[9]}, a3[2] = {p1[8], p1[9]};
        uint32_t af[4] = {*reinterpret_cast<uint32_t *>(a0),
                          *reinterpret_cast<uint32_t *>(a1),
                          *reinterpret_cast<uint32_t *>(a2),
                          *reinterpret_cast<uint32_t *>(a3)};
        mma_m16n8k16(acc[i], af, bf);
      }
    }
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      const int r = rb + i * 16 + qlo;
      const int c = j * 8 + qk;
      sink(r, c + 0, acc[i][0]);
      sink(r, c + 1, acc[i][1]);
      sink(r + 8, c + 0, acc[i][2]);
      sink(r + 8, c + 1, acc[i][3]);
    }
  }
}

__global__ __launch_bounds__(128) void mega_kernel(
    const float *__restrict__ xin, float *__restrict__ xout,
    __half *__restrict__ qkv_scratch,            // [B][SEQ][3D]
    const __half *qkv_w, const __half *qkv_b, const __half *op_w,
    const __half *op_b, const __half *fi_w, const __half *fi_b,
    const float *fo_w, const float *fo_b, const float *fn_w, const float *fn_b,
    int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;

  extern __shared__ __half smem[];
  __half *sK = smem;                 // [SEQ][D]
  __half *sV = sK + SEQ * D;         // [SEQ][D]
  __half *qkvg = qkv_scratch + (size_t)b * SEQ * 3 * D;

  float xr[D];
  const float *xin_row = xin + (size_t)b * SEQ * D + tid * D;
#pragma unroll
  for (int d = 0; d < D; ++d) xr[d] = xin_row[d];

  auto ln_to = [&](__half *dst) {
    float mean = 0.f;
#pragma unroll
    for (int d = 0; d < D; ++d) mean += xr[d];
    mean *= (1.f / D);
    float var = 0.f;
#pragma unroll
    for (int d = 0; d < D; ++d) { float t = xr[d] - mean; var += t * t; }
    var *= (1.f / D);
    float inv = rsqrtf(var + LN_EPS);
#pragma unroll
    for (int d = 0; d < D; ++d)
      dst[tid * D + d] = __float2half((xr[d] - mean) * inv);
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

    // ---- qkv: n1 -> sK, GEMM -> global scratch ----
    ln_to(sK);
    __syncthreads();
    gemm_rowreg<3 * D>(sK, QKVW, warp, lane, [&](int r, int n, float v) {
      qkvg[r * 3 * D + n] = __float2half(v + __half2float(QKVB[n]));
    });
    __threadfence_block();
    __syncthreads();
    // stage k -> sK, v -> sV
#pragma unroll
    for (int d = 0; d < D; ++d) {
      sK[tid * D + d] = qkvg[tid * 3 * D + D + d];
      sV[tid * D + d] = qkvg[tid * 3 * D + 2 * D + d];
    }
    __syncthreads();

    // ---- attention (online fp32 softmax), thread tid = query row ----
    __half ctx[D];
#pragma unroll
    for (int h = 0; h < H; ++h) {
      float m = -CUDART_INF_F, l = 0.f, acc[HD];
#pragma unroll
      for (int e = 0; e < HD; ++e) acc[e] = 0.f;
      const __half *qrow = qkvg + tid * 3 * D + h * HD;   // this thread's q, head h
      for (int j = 0; j <= tid; ++j) {
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < HD; ++e)
          s += __half2float(qrow[e]) * __half2float(sK[j * D + h * HD + e]);
        float mn = fmaxf(m, s);
        float corr = __expf(m - mn), p = __expf(s - mn);
        l = l * corr + p;
#pragma unroll
        for (int e = 0; e < HD; ++e)
          acc[e] = acc[e] * corr + p * __half2float(sV[j * D + h * HD + e]);
        m = mn;
      }
      float linv = 1.f / l;
#pragma unroll
      for (int e = 0; e < HD; ++e) ctx[h * HD + e] = __float2half(acc[e] * linv);
    }
    __syncthreads();

    // ---- out_proj: ctx -> sK, GEMM -> sV (fp16), add to xr ----
#pragma unroll
    for (int d = 0; d < D; ++d) sK[tid * D + d] = ctx[d];
    __syncthreads();
    gemm_rowreg<D>(sK, OPW, warp, lane, [&](int r, int n, float v) {
      sV[r * D + n] = __float2half(v + __half2float(OPB[n]));
    });
    __syncthreads();
#pragma unroll
    for (int d = 0; d < D; ++d) xr[d] += __half2float(sV[tid * D + d]);

    // ---- FFN ----
    ln_to(sK);
    __syncthreads();
    gemm_rowreg<FF>(sK, FIW, warp, lane, [&](int r, int n, float v) {
      sV[r * D + n] = __float2half(v + __half2float(FIB[n]));   // ffn_hidden_fp16
    });
    __syncthreads();
#pragma unroll
    for (int n = 0; n < D; ++n) {
      float a = FOB[n];
#pragma unroll
      for (int k = 0; k < FF; ++k)
        a += gelu_erf(__half2float(sV[tid * D + k])) * FOW[(size_t)n * FF + k];
      xr[n] += a;
    }
    __syncthreads();
  }

  // ---- final_norm (with affine) ----
  float mean = 0.f;
#pragma unroll
  for (int d = 0; d < D; ++d) mean += xr[d];
  mean *= (1.f / D);
  float var = 0.f;
#pragma unroll
  for (int d = 0; d < D; ++d) { float t = xr[d] - mean; var += t * t; }
  var *= (1.f / D);
  float inv = rsqrtf(var + LN_EPS);
  float *orow = xout + (size_t)b * SEQ * D + tid * D;
#pragma unroll
  for (int d = 0; d < D; ++d)
    orow[d] = (xr[d] - mean) * inv * fn_w[d] + fn_b[d];
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
  auto scratch = torch::empty({B, SEQ, 3 * D},
                              x.options().dtype(torch::kHalf));
  size_t smem = (size_t)2 * SEQ * D * sizeof(__half);
  static bool set = false;
  if (!set) {
    cudaFuncSetAttribute(mega_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    set = true;
  }
  mega_kernel<<<B, 128, smem, c10::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), out.data_ptr<float>(),
      (__half *)scratch.data_ptr(),
      (const __half *)qkv_w.data_ptr(), (const __half *)qkv_b.data_ptr(),
      (const __half *)op_w.data_ptr(), (const __half *)op_b.data_ptr(),
      (const __half *)fi_w.data_ptr(), (const __half *)fi_b.data_ptr(),
      fo_w.data_ptr<float>(), fo_b.data_ptr<float>(),
      fn_w.data_ptr<float>(), fn_b.data_ptr<float>(), B);
  C10_CUDA_CHECK(cudaGetLastError());
}
