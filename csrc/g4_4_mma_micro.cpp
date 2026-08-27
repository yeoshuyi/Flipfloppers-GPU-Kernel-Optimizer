// G4.4 Stage 0a -- pybind wrapper for the mma.sync/ldmatrix micro-unit-test.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

void mma_micro_launch(const void *A, const void *W, void *D_ld, void *D_man,
                      cudaStream_t s);

std::vector<torch::Tensor> mma_micro(const torch::Tensor &A,
                                     const torch::Tensor &W) {
  TORCH_CHECK(A.is_cuda() && W.is_cuda(), "cuda only");
  TORCH_CHECK(A.scalar_type() == torch::kHalf && W.scalar_type() == torch::kHalf,
              "fp16 only");
  TORCH_CHECK(A.is_contiguous() && W.is_contiguous(), "contiguous only");
  TORCH_CHECK(A.dim() == 2 && A.size(0) == 16 && A.size(1) == 16, "A must be 16x16");
  TORCH_CHECK(W.dim() == 2 && W.size(0) == 8 && W.size(1) == 16, "W must be 8x16");
  auto D_ld = torch::zeros({16, 8}, A.options());
  auto D_man = torch::zeros({16, 8}, A.options());
  mma_micro_launch(A.data_ptr(), W.data_ptr(), D_ld.data_ptr(),
                   D_man.data_ptr(), c10::cuda::getCurrentCUDAStream());
  C10_CUDA_CHECK(cudaGetLastError());
  return {D_ld, D_man};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mma_micro", &mma_micro, "one-warp 16x16x16 mma.sync fp16-accumulate");
}
