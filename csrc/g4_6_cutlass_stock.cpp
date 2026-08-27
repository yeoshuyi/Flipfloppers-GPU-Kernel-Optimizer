// G4.6 Phase 0 -- pybind wrapper for the stock CUTLASS example-12 config.
// Deliberately mirrors csrc/g4_4_mma_gemm.cpp's shape (thin TORCH_CHECKs, a
// raw-pointer launch declared extern, one PYBIND11_MODULE at the bottom) so
// the two can sit in the same measurement harness.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

int cutlass_stock_launch(const void *A, const void *B, const void *bias,
                         void *D, int M, int N, int K, void *workspace,
                         size_t workspace_bytes, cudaStream_t stream,
                         size_t *workspace_needed);
int cutlass_version_major();
int cutlass_version_minor();
int cutlass_version_patch();

std::string cutlass_version() {
  return std::to_string(cutlass_version_major()) + "." +
         std::to_string(cutlass_version_minor()) + "." +
         std::to_string(cutlass_version_patch());
}

// a: column-major M x K  -> pass a torch tensor of shape (K, M), contiguous
// b: column-major K x N  -> pass a torch tensor of shape (N, K), contiguous
// bias: fp32, length M
// d: column-major M x N  -> pass a torch tensor of shape (N, M), contiguous
void stock_gemm(const torch::Tensor &a_cm, const torch::Tensor &b_cm,
                const torch::Tensor &bias, torch::Tensor &d_cm,
                torch::Tensor &workspace) {
  TORCH_CHECK(a_cm.is_cuda() && b_cm.is_cuda() && d_cm.is_cuda(), "cuda only");
  TORCH_CHECK(a_cm.scalar_type() == torch::kHalf &&
                  b_cm.scalar_type() == torch::kHalf,
              "A/B must be fp16");
  TORCH_CHECK(d_cm.scalar_type() == torch::kFloat &&
                  bias.scalar_type() == torch::kFloat,
              "D/bias must be fp32 (this is the stock example's ElementOutput)");
  TORCH_CHECK(a_cm.is_contiguous() && b_cm.is_contiguous() &&
                  d_cm.is_contiguous() && bias.is_contiguous(),
              "contiguous only");
  const int M = (int)a_cm.size(1), K = (int)a_cm.size(0);
  const int N = (int)b_cm.size(0);
  TORCH_CHECK(b_cm.size(1) == K, "B K mismatch");
  TORCH_CHECK(d_cm.size(0) == N && d_cm.size(1) == M, "D shape mismatch");
  TORCH_CHECK(bias.numel() == M, "bias must be length M");

  size_t need = 0;
  int rc = cutlass_stock_launch(
      a_cm.data_ptr(), b_cm.data_ptr(), bias.data_ptr(), d_cm.data_ptr(), M, N,
      K, workspace.numel() ? workspace.data_ptr() : nullptr,
      (size_t)workspace.numel(), c10::cuda::getCurrentCUDAStream(), &need);
  TORCH_CHECK(rc == 0, "cutlass_stock_launch failed rc=", rc,
              " (-1000 = workspace too small, need=", need,
              "; otherwise -(int)cutlass::Status)");
  C10_CUDA_CHECK(cudaGetLastError());
}

int64_t stock_workspace_bytes(int64_t M, int64_t N, int64_t K) {
  size_t need = 0;
  // A can_implement/get_workspace_size-only probe: pass a zero workspace and
  // read back the requirement through the -1000 path.
  cutlass_stock_launch(nullptr, nullptr, nullptr, nullptr, (int)M, (int)N,
                       (int)K, nullptr, 0, nullptr, &need);
  return (int64_t)need;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_version", &cutlass_version);
  m.def("stock_gemm", &stock_gemm, py::arg("a_cm"), py::arg("b_cm"),
        py::arg("bias"), py::arg("d_cm"), py::arg("workspace"));
  m.def("stock_workspace_bytes", &stock_workspace_bytes, py::arg("M"),
        py::arg("N"), py::arg("K"));
}
