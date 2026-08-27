// G5.MEGA -- per-sequence fused causal transformer megakernel, specialist for
// official row 6 (B=10000, d_model=128, num_heads=4, seq_len=128, layers=4).
//
// ONE CUDA block per sequence; BLOCKDIM = SEQ threads; thread i owns query row i.
// The fp32 residual stream `x` [SEQ][D] lives in shared memory for the entire
// 4-layer forward -- zero HBM round-trips for the residual (step 43: that
// traffic is 38.9% of row 6's forward).
//
// Precision is matched op-for-op to benchmark.py::_optimized_forward_causal so
// this is precision-neutral by construction:
//   * norm1/norm2  : pure (x-mean)*rsqrt(var+eps), fp32   (affine folded into weights)
//   * qkv/out_proj/ffn_in GEMMs : fp16 storage, FP32 accumulate
//   * attention    : fp16 q/k/v upcast to fp32, fp32 softmax w/ max-subtraction,
//                    fp32 P, fp32 PV accumulate    (>= flash accuracy)
//   * ffn hidden   : rounded to fp16 (matches ffn_hidden_fp16), then fp32 GELU(erf)
//   * ffn_out      : fp32 weights, fp32 accumulate
//   * final_norm   : (x-mean)*rsqrt * weight + bias, fp32   (affine kept)
//
// CORRECTNESS-FIRST: scalar loops, no mma.sync. Slow. Phase 1 optimises.

#include <cuda_fp16.h>
#include <math.h>
#include <math_constants.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int D = 128;      // d_model
constexpr int H = 4;        // num_heads
constexpr int HD = D / H;   // head_dim = 32
constexpr int SEQ = 128;
constexpr int NL = 4;       // layers
constexpr int FF = 128;     // ffn_dim
constexpr float LN_EPS = 1e-5f;
constexpr float kAlpha = 0.7071067811865475f;  // 1/sqrt(2), for erf-GELU

__device__ __forceinline__ float gelu_erf(float v) {
  return v * 0.5f * (1.0f + erff(v * kAlpha));
}

// Per-layer weight pointers, all device pointers into the stacked tensors.
struct Layer {
  const __half *qkv_w;  // [3D][D]
  const __half *qkv_b;  // [3D]
  const __half *op_w;   // [D][D]
  const __half *op_b;   // [D]
  const __half *fi_w;   // [FF][D]
  const __half *fi_b;   // [FF]
  const float  *fo_w;   // [D][FF]
  const float  *fo_b;   // [D]
};

__global__ void mega_causal_kernel(const float *__restrict__ xin,   // [B][SEQ][D]
                                   float *__restrict__ xout,        // [B][SEQ][D]
                                   const __half *qkv_w, const __half *qkv_b,
                                   const __half *op_w, const __half *op_b,
                                   const __half *fi_w, const __half *fi_b,
                                   const float *fo_w, const float *fo_b,
                                   const float *fn_w, const float *fn_b,
                                   int B) {
  const int b = blockIdx.x;
  const int i = threadIdx.x;           // query row
  if (b >= B) return;

  extern __shared__ float smem[];
  float *xs = smem;                    // [SEQ][D] fp32   -> SEQ*D*4 bytes
  __half *kh = (__half *)(xs + SEQ * D);  // [SEQ][HD] fp16
  __half *vh = kh + SEQ * HD;             // [SEQ][HD] fp16

  // ---- load residual row ----
  const float *xrow_in = xin + (size_t)b * SEQ * D;
#pragma unroll
  for (int d = 0; d < D; ++d) xs[i * D + d] = xrow_in[i * D + d];
  __syncthreads();

  // per-thread persistent context accumulator across heads (fp16, matches
  // the fp16 `context` the shipped path feeds to out_proj)
  __half ctx[D];

  for (int L = 0; L < NL; ++L) {
    const __half *QKVW = qkv_w + (size_t)L * 3 * D * D;
    const __half *QKVB = qkv_b + (size_t)L * 3 * D;
    const __half *OPW = op_w + (size_t)L * D * D;
    const __half *OPB = op_b + (size_t)L * D;
    const __half *FIW = fi_w + (size_t)L * FF * D;
    const __half *FIB = fi_b + (size_t)L * FF;
    const float *FOW = fo_w + (size_t)L * D * FF;
    const float *FOB = fo_b + (size_t)L * D;

    // ---------- norm1 (no affine) -> n1 in registers (fp16) ----------
    float mean = 0.f;
#pragma unroll
    for (int d = 0; d < D; ++d) mean += xs[i * D + d];
    mean *= (1.f / D);
    float var = 0.f;
#pragma unroll
    for (int d = 0; d < D; ++d) {
      float t = xs[i * D + d] - mean;
      var += t * t;
    }
    var *= (1.f / D);
    float inv = rsqrtf(var + LN_EPS);
    __half n1[D];
#pragma unroll
    for (int d = 0; d < D; ++d)
      n1[d] = __float2half((xs[i * D + d] - mean) * inv);

    // ---------- attention, head by head ----------
    for (int h = 0; h < H; ++h) {
      // q_i / k_i / v_i for this head: [HD] each, from n1 @ QKVW rows
      float qi[HD];
#pragma unroll
      for (int e = 0; e < HD; ++e) {
        int jq = 0 * D + h * HD + e;   // q block
        int jk = 1 * D + h * HD + e;   // k block
        int jv = 2 * D + h * HD + e;   // v block
        float aq = __half2float(QKVB[jq]);
        float ak = __half2float(QKVB[jk]);
        float av = __half2float(QKVB[jv]);
#pragma unroll
        for (int k = 0; k < D; ++k) {
          float nk = __half2float(n1[k]);
          aq += nk * __half2float(QKVW[(size_t)jq * D + k]);
          ak += nk * __half2float(QKVW[(size_t)jk * D + k]);
          av += nk * __half2float(QKVW[(size_t)jv * D + k]);
        }
        qi[e] = aq;
        kh[i * HD + e] = __float2half(ak);
        vh[i * HD + e] = __float2half(av);
      }
      __syncthreads();

      // online softmax over keys j = 0..i (causal), scale = 1.0
      float m = -CUDART_INF_F;
      float l = 0.f;
      float acc[HD];
#pragma unroll
      for (int e = 0; e < HD; ++e) acc[e] = 0.f;
      for (int j = 0; j <= i; ++j) {
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < HD; ++e)
          s += qi[e] * __half2float(kh[j * HD + e]);
        float m_new = fmaxf(m, s);
        float corr = __expf(m - m_new);
        float p = __expf(s - m_new);
        l = l * corr + p;
#pragma unroll
        for (int e = 0; e < HD; ++e)
          acc[e] = acc[e] * corr + p * __half2float(vh[j * HD + e]);
        m = m_new;
      }
      float linv = 1.f / l;
#pragma unroll
      for (int e = 0; e < HD; ++e)
        ctx[h * HD + e] = __float2half(acc[e] * linv);
      __syncthreads();   // before next head overwrites kh/vh
    }

    // ---------- out_proj + residual (row-local) ----------
#pragma unroll
    for (int jo = 0; jo < D; ++jo) {
      float a = __half2float(OPB[jo]);
#pragma unroll
      for (int k = 0; k < D; ++k)
        a += __half2float(ctx[k]) * __half2float(OPW[(size_t)jo * D + k]);
      xs[i * D + jo] += a;
    }

    // ---------- norm2 (no affine) -> n2 in registers ----------
    mean = 0.f;
#pragma unroll
    for (int d = 0; d < D; ++d) mean += xs[i * D + d];
    mean *= (1.f / D);
    var = 0.f;
#pragma unroll
    for (int d = 0; d < D; ++d) {
      float t = xs[i * D + d] - mean;
      var += t * t;
    }
    var *= (1.f / D);
    inv = rsqrtf(var + LN_EPS);
    __half n2[D];
#pragma unroll
    for (int d = 0; d < D; ++d)
      n2[d] = __float2half((xs[i * D + d] - mean) * inv);

    // ---------- ffn_in (fp32 accum -> fp16 hidden) + GELU ----------
    float g[FF];
#pragma unroll
    for (int f = 0; f < FF; ++f) {
      float a = __half2float(FIB[f]);
#pragma unroll
      for (int k = 0; k < D; ++k)
        a += __half2float(n2[k]) * __half2float(FIW[(size_t)f * D + k]);
      g[f] = gelu_erf(__half2float(__float2half(a)));  // round to fp16 then erf-GELU in fp32
    }

    // ---------- ffn_out (fp32) + residual ----------
#pragma unroll
    for (int jo = 0; jo < D; ++jo) {
      float a = FOB[jo];
#pragma unroll
      for (int k = 0; k < FF; ++k)
        a += g[k] * FOW[(size_t)jo * FF + k];
      xs[i * D + jo] += a;
    }
    __syncthreads();
  }

  // ---------- final_norm (WITH affine) -> output ----------
  float mean = 0.f;
#pragma unroll
  for (int d = 0; d < D; ++d) mean += xs[i * D + d];
  mean *= (1.f / D);
  float var = 0.f;
#pragma unroll
  for (int d = 0; d < D; ++d) {
    float t = xs[i * D + d] - mean;
    var += t * t;
  }
  var *= (1.f / D);
  float inv = rsqrtf(var + LN_EPS);
  float *orow = xout + (size_t)b * SEQ * D;
#pragma unroll
  for (int d = 0; d < D; ++d)
    orow[i * D + d] = (xs[i * D + d] - mean) * inv * fn_w[d] + fn_b[d];
}

}  // namespace

// ---------------------------------------------------------------------------
// entry: everything pre-checked on the Python side (row-6 specialist).
void mega_causal_forward(const torch::Tensor &x,          // [B,SEQ,D] fp32
                         torch::Tensor &out,              // [B,SEQ,D] fp32
                         const torch::Tensor &qkv_w,      // [NL,3D,D] fp16
                         const torch::Tensor &qkv_b,      // [NL,3D]  fp16
                         const torch::Tensor &op_w,       // [NL,D,D] fp16
                         const torch::Tensor &op_b,       // [NL,D]   fp16
                         const torch::Tensor &fi_w,       // [NL,FF,D] fp16
                         const torch::Tensor &fi_b,       // [NL,FF]  fp16
                         const torch::Tensor &fo_w,       // [NL,D,FF] fp32
                         const torch::Tensor &fo_b,       // [NL,D]   fp32
                         const torch::Tensor &fn_w,       // [D] fp32
                         const torch::Tensor &fn_b) {     // [D] fp32
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat, "x f32 cuda");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == SEQ && x.size(2) == D, "x shape");
  TORCH_CHECK(out.sizes() == x.sizes() && out.scalar_type() == torch::kFloat, "out");
  const int B = (int)x.size(0);

  size_t smem = (size_t)SEQ * D * sizeof(float) + (size_t)2 * SEQ * HD * sizeof(__half);
  static bool set = false;
  if (!set) {
    cudaFuncSetAttribute(mega_causal_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    set = true;
  }
  mega_causal_kernel<<<B, SEQ, smem, c10::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), out.data_ptr<float>(),
      (const __half *)qkv_w.data_ptr(), (const __half *)qkv_b.data_ptr(),
      (const __half *)op_w.data_ptr(), (const __half *)op_b.data_ptr(),
      (const __half *)fi_w.data_ptr(), (const __half *)fi_b.data_ptr(),
      fo_w.data_ptr<float>(), fo_b.data_ptr<float>(),
      fn_w.data_ptr<float>(), fn_b.data_ptr<float>(), B);
  C10_CUDA_CHECK(cudaGetLastError());
}
