// G4.6 Phase 1 -- pybind dispatcher over the per-TU CUTLASS instantiations.
// Deliberately mirrors csrc/g4_4_mma_gemm.cpp so the two kernels can be swapped
// inside the same measurement harness (probes/g4_4_mma_gemm_stage0c.py's
// graph_time / prof_kernel_us) with nothing else changing.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <sstream>
#include <string>

using LaunchFn = int (*)(const void *, const void *, const void *, void *, int,
                         int, int, void *, size_t, cudaStream_t, size_t *,
                         int *);

#define DECL(i)                                                              \
  int g46_launch_##i(const void *, const void *, const void *, void *, int,  \
                     int, int, void *, size_t, cudaStream_t, size_t *, int *);
// ---- BEGIN GENERATED kCfg (probes/g4_6_gen_cfgs.py) ----
DECL(0)
DECL(1)
DECL(2)
DECL(3)
DECL(4)
DECL(5)
DECL(6)
DECL(7)
DECL(8)
DECL(9)
DECL(10)
DECL(11)
DECL(12)
DECL(13)
DECL(14)
DECL(15)
DECL(16)
DECL(17)
DECL(18)
DECL(19)
DECL(20)
DECL(21)
DECL(22)
DECL(23)
#undef DECL

namespace {
struct CfgDesc {
  LaunchFn fn;
  int BM, BN, BK, WM, WN, WK, STG, SWZ, ACCF32;
};
const CfgDesc kCfg[] = {
    {g46_launch_0, 128, 128, 64, 64, 64, 64, 3, 1, 0},
    {g46_launch_1, 128, 128, 64, 64, 64, 64, 2, 1, 0},
    {g46_launch_2, 128, 128, 32, 64, 64, 32, 4, 1, 0},
    {g46_launch_3, 128, 128, 32, 64, 64, 32, 6, 1, 0},
    {g46_launch_4, 128, 256, 64, 64, 64, 64, 2, 1, 0},
    {g46_launch_5, 256, 128, 64, 64, 64, 64, 2, 1, 0},
    {g46_launch_6, 128, 256, 32, 64, 64, 32, 4, 1, 0},
    {g46_launch_7, 256, 128, 32, 64, 64, 32, 4, 1, 0},
    {g46_launch_8, 128, 64, 64, 64, 32, 64, 4, 1, 0},
    {g46_launch_9, 64, 128, 64, 32, 64, 64, 4, 1, 0},
    {g46_launch_10, 128, 128, 64, 64, 64, 64, 3, 1, 1},
    {g46_launch_11, 128, 128, 64, 64, 64, 64, 2, 1, 1},
    {g46_launch_12, 128, 128, 32, 64, 64, 32, 4, 1, 1},
    {g46_launch_13, 128, 256, 64, 64, 64, 64, 2, 1, 1},
    {g46_launch_14, 128, 256, 32, 64, 64, 32, 4, 4, 0},
    {g46_launch_15, 128, 256, 32, 64, 64, 32, 4, 8, 0},
    {g46_launch_16, 256, 128, 32, 64, 64, 32, 4, 4, 0},
    {g46_launch_17, 128, 256, 32, 64, 64, 32, 3, 1, 0},
    {g46_launch_18, 128, 256, 32, 64, 64, 32, 2, 1, 0},
    {g46_launch_19, 256, 256, 32, 64, 64, 32, 3, 1, 0},
    {g46_launch_20, 64, 256, 64, 32, 64, 64, 2, 1, 0},
    {g46_launch_21, 128, 256, 64, 64, 64, 64, 2, 4, 0},
    {g46_launch_22, 256, 128, 64, 64, 64, 64, 2, 4, 0},
    {g46_launch_23, 128, 128, 64, 64, 64, 64, 3, 4, 0},
};
constexpr int kNumCfg = (int)(sizeof(kCfg) / sizeof(kCfg[0]));
}  // namespace
// ---- END GENERATED kCfg ----

int64_t num_cfg() { return kNumCfg; }

std::string cfg_name(int64_t c) {
  TORCH_CHECK(c >= 0 && c < kNumCfg, "bad cfg");
  const CfgDesc &d = kCfg[c];
  int smem = 0;
  size_t need = 0;
  d.fn(nullptr, nullptr, nullptr, nullptr, 128, 128, 128, nullptr, 0, nullptr,
       &need, &smem);
  std::ostringstream o;
  o << (d.ACCF32 ? "accF32 " : "accF16 ") << "TB" << d.BM << "x" << d.BN << "x"
    << d.BK << " W" << d.WM << "x" << d.WN << "x" << d.WK << " stg" << d.STG
    << " swz" << d.SWZ << " smem=" << (smem / 1024) << "KB";
  return o.str();
}

int64_t cfg_smem_bytes(int64_t c) {
  TORCH_CHECK(c >= 0 && c < kNumCfg, "bad cfg");
  int smem = 0;
  size_t need = 0;
  kCfg[c].fn(nullptr, nullptr, nullptr, nullptr, 128, 128, 128, nullptr, 0,
             nullptr, &need, &smem);
  return smem;
}

int64_t cfg_workspace_bytes(int64_t c, int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(c >= 0 && c < kNumCfg, "bad cfg");
  int smem = 0;
  size_t need = 0;
  kCfg[c].fn(nullptr, nullptr, nullptr, nullptr, (int)M, (int)N, (int)K,
             nullptr, 0, nullptr, &need, &smem);
  return (int64_t)need;
}

// out[M,N] = inp[M,K] @ w[N,K]^T + bias[N]   -- i.e. F.linear, all fp16.
void cutlass_gemm(int64_t cfg, const torch::Tensor &inp, const torch::Tensor &w,
                  const torch::Tensor &bias, torch::Tensor &out,
                  torch::Tensor &workspace) {
  TORCH_CHECK(cfg >= 0 && cfg < kNumCfg, "bad cfg");
  TORCH_CHECK(inp.is_cuda() && w.is_cuda() && out.is_cuda(), "cuda only");
  TORCH_CHECK(inp.scalar_type() == torch::kHalf &&
                  w.scalar_type() == torch::kHalf &&
                  out.scalar_type() == torch::kHalf &&
                  bias.scalar_type() == torch::kHalf,
              "fp16 only");
  TORCH_CHECK(inp.is_contiguous() && w.is_contiguous() && out.is_contiguous() &&
                  bias.is_contiguous(),
              "contiguous only");
  TORCH_CHECK(inp.dim() == 2 && w.dim() == 2 && out.dim() == 2, "2-D only");
  const int M = (int)inp.size(0), K = (int)inp.size(1), N = (int)w.size(0);
  TORCH_CHECK(w.size(1) == K, "w K mismatch");
  TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out shape mismatch");
  TORCH_CHECK(bias.numel() == N, "bias must be length N");

  size_t need = 0;
  int smem = 0;
  int rc = kCfg[cfg].fn(inp.data_ptr(), w.data_ptr(), bias.data_ptr(),
                        out.data_ptr(), M, N, K,
                        workspace.numel() ? workspace.data_ptr() : nullptr,
                        (size_t)workspace.numel(),
                        c10::cuda::getCurrentCUDAStream(), &need, &smem);
  TORCH_CHECK(rc == 0, "g46_launch failed rc=", rc, " (-1000 = workspace too "
              "small, need=", need, "; otherwise -(int)cutlass::Status: "
              "1=kErrorMisalignedOperand 2=kErrorInvalidDataType "
              "3=kErrorInvalidLayout 4=kErrorInvalidProblem "
              "5=kErrorNotSupported 6=kErrorWorkspaceNull "
              "7=kErrorInternal ... see cutlass/cutlass.h)");
  C10_CUDA_CHECK(cudaGetLastError());
}

torch::Tensor cutlass_linear(int64_t cfg, const torch::Tensor &inp,
                             const torch::Tensor &w, const torch::Tensor &bias,
                             torch::Tensor &workspace) {
  auto out = torch::empty({inp.size(0), w.size(0)}, inp.options());
  cutlass_gemm(cfg, inp, w, bias, out, workspace);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("num_cfg", &num_cfg);
  m.def("cfg_name", &cfg_name, py::arg("cfg"));
  m.def("cfg_smem_bytes", &cfg_smem_bytes, py::arg("cfg"));
  m.def("cfg_workspace_bytes", &cfg_workspace_bytes, py::arg("cfg"),
        py::arg("M"), py::arg("N"), py::arg("K"));
  m.def("cutlass_gemm", &cutlass_gemm, py::arg("cfg"), py::arg("inp"),
        py::arg("w"), py::arg("bias"), py::arg("out"), py::arg("workspace"));
  m.def("cutlass_linear", &cutlass_linear, py::arg("cfg"), py::arg("inp"),
        py::arg("w"), py::arg("bias"), py::arg("workspace"));
}
