// G4.4 -- pybind wrapper for the hand-written mma.sync FP16-accumulate GEMM.
// Calling convention deliberately mirrors csrc/cublaslt_algo_fp16.cpp's
// run()/lt_linear() so the two can be swapped inside the same measurement
// harness with nothing else changing.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <sstream>
#include <string>

int mma_gemm_launch(int cfg, const void *In, const void *W, const void *bias,
                    void *Out, int M, int N, int K, cudaStream_t s,
                    size_t *smem_out);
int mma_gemm_num_cfg();

namespace {
struct CfgDesc { int BM, BN, BK, NSTAGE, SPLIT, ACCF32; };
const CfgDesc kCfg[] = {
    {64, 128, 64, 3, 0, 0},   {64, 128, 64, 2, 0, 0},
    {64, 128, 32, 7, 0, 0},   {64, 128, 32, 4, 0, 0},
    {64, 128, 32, 5, 0, 0},   {64, 128, 64, 3, 256, 0},
    {64, 128, 32, 7, 256, 0}, {64, 128, 64, 3, 128, 0},
    {64, 128, 32, 7, 128, 0}, {64, 128, 64, 3, 64, 0},
    {128, 128, 64, 3, 0, 0},  {128, 128, 64, 2, 0, 0},
    {128, 128, 32, 6, 0, 0},  {128, 128, 32, 4, 0, 0},
    {128, 128, 64, 3, 256, 0},
    {64, 128, 64, 3, 0, 1},   {64, 128, 64, 2, 0, 1},
    {128, 128, 64, 3, 0, 1},  {128, 128, 64, 2, 0, 1},
    {128, 256, 64, 2, 0, 0},  {256, 128, 32, 4, 0, 0},
    {128, 256, 32, 4, 0, 0},  {256, 128, 64, 2, 0, 0},
    {128, 256, 64, 2, 0, 1},  {256, 128, 64, 2, 0, 1},
    {128, 256, 64, 2, 256, 0},
};
}  // namespace

int64_t num_cfg() { return mma_gemm_num_cfg(); }

std::string cfg_name(int64_t c) {
  TORCH_CHECK(c >= 0 && c < mma_gemm_num_cfg(), "bad cfg");
  const CfgDesc &d = kCfg[c];
  const size_t smem = (size_t)d.NSTAGE * (d.BM * d.BK + d.BN * d.BK) * 2;
  std::ostringstream o;
  o << (d.ACCF32 ? "accF32 " : "accF16 ") << "BM" << d.BM << " BN" << d.BN
    << " BK" << d.BK << " stg" << d.NSTAGE << " split" << d.SPLIT
    << " smem=" << (smem / 1024) << "KB";
  return o.str();
}

void mma_gemm(int64_t cfg, const torch::Tensor &inp, const torch::Tensor &w,
              const c10::optional<torch::Tensor> &bias, torch::Tensor &out) {
  TORCH_CHECK(inp.is_cuda() && w.is_cuda() && out.is_cuda(), "cuda only");
  TORCH_CHECK(inp.scalar_type() == torch::kHalf &&
                  w.scalar_type() == torch::kHalf &&
                  out.scalar_type() == torch::kHalf, "fp16 only");
  TORCH_CHECK(inp.is_contiguous() && w.is_contiguous() && out.is_contiguous(),
              "contiguous only");
  TORCH_CHECK(inp.dim() == 2 && w.dim() == 2 && out.dim() == 2, "2-D only");
  const int M = (int)inp.size(0), K = (int)inp.size(1), N = (int)w.size(0);
  TORCH_CHECK(w.size(1) == K, "w K mismatch");
  TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out shape mismatch");
  const void *bp = nullptr;
  if (bias.has_value()) {
    const torch::Tensor &b = bias.value();
    TORCH_CHECK(b.is_cuda() && b.scalar_type() == torch::kHalf &&
                    b.is_contiguous() && b.numel() == N, "bad bias");
    bp = b.data_ptr();
  }
  size_t smem = 0;
  int rc = mma_gemm_launch((int)cfg, inp.data_ptr(), w.data_ptr(), bp,
                           out.data_ptr(), M, N, K,
                           c10::cuda::getCurrentCUDAStream(), &smem);
  TORCH_CHECK(rc == 0, "mma_gemm_launch failed rc=", rc,
              " (-2 = shape not divisible by tile, -3 = K%SPLIT, "
              "-4 = smem opt-in rejected, smem=", smem, ")");
  C10_CUDA_CHECK(cudaGetLastError());
}

torch::Tensor mma_linear(int64_t cfg, const torch::Tensor &inp,
                         const torch::Tensor &w,
                         const c10::optional<torch::Tensor> &bias) {
  auto out = torch::empty({inp.size(0), w.size(0)}, inp.options());
  mma_gemm(cfg, inp, w, bias, out);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("num_cfg", &num_cfg);
  m.def("cfg_name", &cfg_name, py::arg("cfg"));
  m.def("mma_gemm", &mma_gemm, py::arg("cfg"), py::arg("inp"), py::arg("w"),
        py::arg("bias"), py::arg("out"));
  m.def("mma_linear", &mma_linear, py::arg("cfg"), py::arg("inp"),
        py::arg("w"), py::arg("bias"));
}
