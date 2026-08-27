// G4.3 -- pybind wrapper for the warp-specialised mma.sync FP16-accumulate
// GEMM.  Calling convention is IDENTICAL to csrc/g4_4_mma_gemm.cpp's
// mma_gemm()/mma_linear() so the two can be swapped inside the same
// measurement harness with nothing else changing (which is what makes
// warpspec cfg[0] a valid bitwise control for g4_4 cfg[11]).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

int ws_gemm_launch(int cfg, const void *In, const void *W, const void *bias,
                   void *Out, int M, int N, int K, cudaStream_t s,
                   size_t *smem_out);
int ws_gemm_num_cfg();
void ws_gemm_cfg_desc(int cfg, char *buf, int buflen);

int64_t num_cfg() { return ws_gemm_num_cfg(); }

std::string cfg_name(int64_t c) {
  TORCH_CHECK(c >= 0 && c < ws_gemm_num_cfg(), "bad cfg");
  std::vector<char> buf(256, 0);
  ws_gemm_cfg_desc((int)c, buf.data(), (int)buf.size());
  return std::string(buf.data());
}

void ws_gemm(int64_t cfg, const torch::Tensor &inp, const torch::Tensor &w,
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
  int rc = ws_gemm_launch((int)cfg, inp.data_ptr(), w.data_ptr(), bp,
                          out.data_ptr(), M, N, K,
                          c10::cuda::getCurrentCUDAStream(), &smem);
  TORCH_CHECK(rc == 0, "ws_gemm_launch failed rc=", rc,
              " (-2 = shape not divisible by tile, -3 = K%SPLIT, "
              "-4 = smem opt-in rejected, smem=", smem, ")");
  C10_CUDA_CHECK(cudaGetLastError());
}

torch::Tensor ws_linear(int64_t cfg, const torch::Tensor &inp,
                        const torch::Tensor &w,
                        const c10::optional<torch::Tensor> &bias) {
  auto out = torch::empty({inp.size(0), w.size(0)}, inp.options());
  ws_gemm(cfg, inp, w, bias, out);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("num_cfg", &num_cfg);
  m.def("cfg_name", &cfg_name, py::arg("cfg"));
  m.def("ws_gemm", &ws_gemm, py::arg("cfg"), py::arg("inp"), py::arg("w"),
        py::arg("bias"), py::arg("out"));
  m.def("ws_linear", &ws_linear, py::arg("cfg"), py::arg("inp"), py::arg("w"),
        py::arg("bias"));
}
