// G6.6 -- raw cuBLASLt matmul with explicit algorithm selection.
//
// PURPOSE (step 1, standalone probe): step 31's ncu pass found the FFN's TF32
// CUTLASS GEMM (cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4) running at
// only ~47% of TF32 peak with ~26% occupancy. Step 25 closed the obvious lever
// (torch.compile max-autotune) on ACCURACY, because inductor's search mixes in
// Triton GEMMs that emulate TF32 with a 3-pass FP32 decomposition -- a
// different reduction algorithm than cuBLAS's native TF32 tensor-core path.
//
// This extension tests a narrower hypothesis with a different risk profile:
// cuBLASLt exposes several NATIVE algorithm variants (tile size, split-K,
// stages, CTA swizzle) for the same shape/dtype, all on the same tensor-core
// datapath PyTorch already uses. If one of them beats the default heuristic
// pick for our two fixed FFN shapes, that is a win with no new numerics family.
//
// LAYOUT NOTE -- this is the part that is easy to get wrong.
// PyTorch is row-major. F.linear computes Out[M,N] = In[M,K] @ W[N,K]^T.
// cuBLASLt is column-major. A row-major [r,c] buffer with ld=c IS a
// column-major [c,r] buffer with ld=c. So:
//     Out row-major [M,N]  ==  column-major [N,M], ld = N
//     In  row-major [M,K]  ==  column-major [K,M], ld = K
//     W   row-major [N,K]  ==  column-major [K,N], ld = K
// and Out_cm[N,M] = (W_cm)^T[N,K] * In_cm[K,M], i.e. a "TN" gemm with
//     m = N, n = M, k = K,  A = W (transa=T, stored K x N), B = In (transb=N).
// The "_tn_" in the profiled kernel name is the same configuration, which is
// the cheap confirmation that this mapping matches what PyTorch itself emits.
//
// Build: torch.utils.cpp_extension.load(..., with_cuda=True,
//        extra_ldflags=["-lcublasLt"]).  with_cuda=True is mandatory even with
//        no .cu source (docs/PROGRESS.md step 32).

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

inline void ckc(cublasStatus_t s, const char *what) {
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, what, " failed, status=", (int)s);
}
inline void ck(cudaError_t e, const char *what) {
  TORCH_CHECK(e == cudaSuccess, what, " failed: ", cudaGetErrorString(e));
}

cublasLtHandle_t lt_handle() {
  static cublasLtHandle_t h = nullptr;
  if (h == nullptr) ckc(cublasLtCreate(&h), "cublasLtCreate");
  return h;
}

// One fixed GEMM problem: fixed M/N/K, fixed epilogue, plus the list of
// candidate algorithms the heuristic returned for it.
struct Problem {
  int64_t M = 0, N = 0, K = 0;
  bool use_bias = false;
  size_t max_ws = 0;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr;  // W : K x N, ld K   (transa = T)
  cublasLtMatrixLayout_t Bdesc = nullptr;  // In: K x M, ld K   (transb = N)
  cublasLtMatrixLayout_t Cdesc = nullptr;  // Out: N x M, ld N
  std::vector<cublasLtMatmulHeuristicResult_t> algos;
  torch::Tensor workspace;

  ~Problem() {
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

std::vector<std::shared_ptr<Problem>> g_problems;

Problem &get(int64_t pid) {
  TORCH_CHECK(pid >= 0 && pid < (int64_t)g_problems.size(), "bad problem id");
  return *g_problems[pid];
}

int32_t cfg_i32(const cublasLtMatmulAlgo_t &a, cublasLtMatmulAlgoConfigAttributes_t attr) {
  int32_t v = -1;
  size_t written = 0;
  if (cublasLtMatmulAlgoConfigGetAttribute(&a, attr, &v, sizeof(v), &written) !=
      CUBLAS_STATUS_SUCCESS) {
    return -1;
  }
  return v;
}

}  // namespace

// Creates the descriptors for one (M,N,K) F.linear-shaped TF32 problem and runs
// cublasLtMatmulAlgoGetHeuristic for `requested` candidates. Returns the problem
// id; the number of candidates actually returned is num_algos(pid).
int64_t create_problem(int64_t M, int64_t N, int64_t K, bool use_bias,
                       int64_t max_ws_bytes, int64_t requested) {
  TORCH_CHECK(M > 0 && N > 0 && K > 0, "bad dims");
  TORCH_CHECK(requested > 0 && requested <= 64, "requested out of range");

  auto p = std::make_shared<Problem>();
  p->M = M; p->N = N; p->K = K; p->use_bias = use_bias;
  p->max_ws = (size_t)max_ws_bytes;

  ckc(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F),
      "cublasLtMatmulDescCreate");
  cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
  ckc(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &opT,
                                     sizeof(opT)), "set TRANSA");
  ckc(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &opN,
                                     sizeof(opN)), "set TRANSB");
  if (use_bias) {
    cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
    ckc(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep,
                                       sizeof(ep)), "set EPILOGUE");
  }

  ckc(cublasLtMatrixLayoutCreate(&p->Adesc, CUDA_R_32F, K, N, K), "layout A");
  ckc(cublasLtMatrixLayoutCreate(&p->Bdesc, CUDA_R_32F, K, M, K), "layout B");
  ckc(cublasLtMatrixLayoutCreate(&p->Cdesc, CUDA_R_32F, N, M, N), "layout C");

  cublasLtMatmulPreference_t pref = nullptr;
  ckc(cublasLtMatmulPreferenceCreate(&pref), "prefCreate");
  size_t ws = (size_t)max_ws_bytes;
  ckc(cublasLtMatmulPreferenceSetAttribute(
          pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)),
      "pref workspace");

  std::vector<cublasLtMatmulHeuristicResult_t> res((size_t)requested);
  int returned = 0;
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
      lt_handle(), p->op, p->Adesc, p->Bdesc, p->Cdesc, p->Cdesc, pref,
      (int)requested, res.data(), &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  ckc(st, "cublasLtMatmulAlgoGetHeuristic");

  res.resize((size_t)returned);
  p->algos = std::move(res);

  if (max_ws_bytes > 0) {
    p->workspace = torch::empty({max_ws_bytes},
                                torch::dtype(torch::kUInt8).device(torch::kCUDA));
  }

  g_problems.push_back(p);
  return (int64_t)g_problems.size() - 1;
}

int64_t num_algos(int64_t pid) { return (int64_t)get(pid).algos.size(); }

// Human-readable config of one candidate, so a "winner" can be reported as a
// concrete algorithm rather than an opaque index.
std::string algo_info(int64_t pid, int64_t idx) {
  Problem &p = get(pid);
  TORCH_CHECK(idx >= 0 && idx < (int64_t)p.algos.size(), "bad algo idx");
  const cublasLtMatmulAlgo_t &a = p.algos[idx].algo;
  std::ostringstream o;
  o << "id=" << cfg_i32(a, CUBLASLT_ALGO_CONFIG_ID)
    << " tile=" << cfg_i32(a, CUBLASLT_ALGO_CONFIG_TILE_ID)
    << " stages=" << cfg_i32(a, CUBLASLT_ALGO_CONFIG_STAGES_ID)
    << " splitk=" << cfg_i32(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
    << " reduc=" << cfg_i32(a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)
    << " swizzle=" << cfg_i32(a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)
    << " ws=" << (int64_t)p.algos[idx].workspaceSize
    << " wave=" << p.algos[idx].wavesCount;
  return o.str();
}

// out = inp @ w^T (+ bias). inp [M,K], w [N,K], bias [N] or undefined, out [M,N],
// all row-major contiguous float32.
void run(int64_t pid, int64_t idx, const torch::Tensor &inp,
         const torch::Tensor &w, const c10::optional<torch::Tensor> &bias,
         torch::Tensor &out) {
  Problem &p = get(pid);
  TORCH_CHECK(idx >= 0 && idx < (int64_t)p.algos.size(), "bad algo idx");
  TORCH_CHECK(inp.is_cuda() && w.is_cuda() && out.is_cuda(), "cuda tensors only");
  TORCH_CHECK(inp.scalar_type() == torch::kFloat32 &&
              w.scalar_type() == torch::kFloat32 &&
              out.scalar_type() == torch::kFloat32, "float32 only");
  TORCH_CHECK(inp.is_contiguous() && w.is_contiguous() && out.is_contiguous(),
              "contiguous only");
  TORCH_CHECK(inp.dim() == 2 && inp.size(0) == p.M && inp.size(1) == p.K,
              "inp shape mismatch");
  TORCH_CHECK(w.dim() == 2 && w.size(0) == p.N && w.size(1) == p.K,
              "w shape mismatch");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == p.M && out.size(1) == p.N,
              "out shape mismatch");

  if (p.use_bias) {
    TORCH_CHECK(bias.has_value(), "problem was created with bias epilogue");
    const torch::Tensor &b = bias.value();
    TORCH_CHECK(b.is_cuda() && b.scalar_type() == torch::kFloat32 &&
                b.is_contiguous() && b.numel() == p.N, "bad bias");
    const void *bp = b.data_ptr();
    ckc(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                       &bp, sizeof(bp)), "set BIAS_POINTER");
  }

  const float alpha = 1.0f, beta = 0.0f;
  void *wsptr = p.workspace.defined() ? p.workspace.data_ptr() : nullptr;
  size_t wsbytes = p.workspace.defined() ? p.max_ws : 0;
  cudaStream_t s = c10::cuda::getCurrentCUDAStream();

  ckc(cublasLtMatmul(lt_handle(), p.op, &alpha, w.data_ptr(), p.Adesc,
                     inp.data_ptr(), p.Bdesc, &beta, out.data_ptr(), p.Cdesc,
                     out.data_ptr(), p.Cdesc, &p.algos[idx].algo, wsptr, wsbytes,
                     s),
      "cublasLtMatmul");
}

// Allocating form of run(), for use behind a torch.library custom op: returns a
// fresh [M,N] tensor instead of writing into a caller-supplied one. Allocation
// goes through torch's caching allocator, so under CUDA-graph capture it comes
// from the graph's own pool and inductor's cudagraph-trees bookkeeping sees a
// normal graph-owned output -- which an out-parameter mutated by an opaque
// extern call would not give it.
torch::Tensor lt_linear(int64_t pid, int64_t idx, const torch::Tensor &inp,
                        const torch::Tensor &w,
                        const c10::optional<torch::Tensor> &bias) {
  Problem &p = get(pid);  // also validates pid before any allocation
  torch::Tensor out = torch::empty({inp.size(0), p.N}, inp.options());
  run(pid, idx, inp, w, bias, out);
  return out;
}

// Times `iters` back-to-back launches of one candidate with CUDA events, on the
// current stream, after `warmup` untimed launches. Returns mean ms per call.
// Timing lives in C++ so no Python dispatch overhead sits inside the measured
// region -- the GEMM is ~50us so it would not dominate, but this removes the
// question entirely.
double time_algo(int64_t pid, int64_t idx, const torch::Tensor &inp,
                 const torch::Tensor &w, const c10::optional<torch::Tensor> &bias,
                 torch::Tensor &out, int64_t warmup, int64_t iters) {
  TORCH_CHECK(iters > 0, "iters must be positive");
  for (int64_t i = 0; i < warmup; ++i) run(pid, idx, inp, w, bias, out);
  cudaStream_t s = c10::cuda::getCurrentCUDAStream();
  ck(cudaStreamSynchronize(s), "sync before timing");

  cudaEvent_t e0, e1;
  ck(cudaEventCreate(&e0), "eventCreate");
  ck(cudaEventCreate(&e1), "eventCreate");
  ck(cudaEventRecord(e0, s), "eventRecord");
  for (int64_t i = 0; i < iters; ++i) run(pid, idx, inp, w, bias, out);
  ck(cudaEventRecord(e1, s), "eventRecord");
  ck(cudaEventSynchronize(e1), "eventSynchronize");
  float ms = 0.f;
  ck(cudaEventElapsedTime(&ms, e0, e1), "eventElapsedTime");
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return (double)ms / (double)iters;
}

// Same loop as time_algo, but ALSO returns the CPU wall time spent issuing the
// launches. If cpu_ms ~= gpu_ms the loop is launch-bound and the "GPU time" is
// really the CPU's issue rate -- which matters at M=64, where the GEMM itself is
// only ~130 MFLOP and any measured difference could be dispatch cost rather than
// kernel cost. Returns {gpu_ms_per_iter, cpu_ms_per_iter}.
std::vector<double> time_algo2(int64_t pid, int64_t idx, const torch::Tensor &inp,
                               const torch::Tensor &w,
                               const c10::optional<torch::Tensor> &bias,
                               torch::Tensor &out, int64_t warmup, int64_t iters) {
  TORCH_CHECK(iters > 0, "iters must be positive");
  for (int64_t i = 0; i < warmup; ++i) run(pid, idx, inp, w, bias, out);
  cudaStream_t s = c10::cuda::getCurrentCUDAStream();
  ck(cudaStreamSynchronize(s), "sync before timing");

  cudaEvent_t e0, e1;
  ck(cudaEventCreate(&e0), "eventCreate");
  ck(cudaEventCreate(&e1), "eventCreate");
  auto c0 = std::chrono::steady_clock::now();
  ck(cudaEventRecord(e0, s), "eventRecord");
  for (int64_t i = 0; i < iters; ++i) run(pid, idx, inp, w, bias, out);
  ck(cudaEventRecord(e1, s), "eventRecord");
  auto c1 = std::chrono::steady_clock::now();  // BEFORE the sync: issue time only
  ck(cudaEventSynchronize(e1), "eventSynchronize");
  float ms = 0.f;
  ck(cudaEventElapsedTime(&ms, e0, e1), "eventElapsedTime");
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  double cpu_ms =
      std::chrono::duration<double, std::milli>(c1 - c0).count();
  return {(double)ms / (double)iters, cpu_ms / (double)iters};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_problem", &create_problem,
        "create a TF32 F.linear-shaped cuBLASLt problem + run algo heuristic",
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("use_bias"),
        py::arg("max_ws_bytes"), py::arg("requested"));
  m.def("num_algos", &num_algos, "candidates returned by the heuristic");
  m.def("algo_info", &algo_info, "describe one candidate algorithm");
  m.def("run", &run, "out = inp @ w^T (+bias) with a chosen algorithm",
        py::arg("pid"), py::arg("idx"), py::arg("inp"), py::arg("w"),
        py::arg("bias"), py::arg("out"));
  m.def("lt_linear", &lt_linear, "allocating form of run(), returns [M,N]",
        py::arg("pid"), py::arg("idx"), py::arg("inp"), py::arg("w"),
        py::arg("bias"));
  m.def("time_algo", &time_algo, "CUDA-event mean ms/call for one candidate",
        py::arg("pid"), py::arg("idx"), py::arg("inp"), py::arg("w"),
        py::arg("bias"), py::arg("out"), py::arg("warmup"), py::arg("iters"));
  m.def("time_algo2", &time_algo2, "{gpu_ms, cpu_issue_ms} per call",
        py::arg("pid"), py::arg("idx"), py::arg("inp"), py::arg("w"),
        py::arg("bias"), py::arg("out"), py::arg("warmup"), py::arg("iters"));
}
