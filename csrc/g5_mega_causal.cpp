// G5.MEGA -- pybind wrapper for the per-sequence fused causal megakernel
// (row-6 specialist). All shape/dtype validation is on the Python side; the
// kernel is hard-specialised for d_model=128, heads=4, seq_len=128, layers=4.

#include <torch/extension.h>

void mega_causal_forward(const torch::Tensor &x, torch::Tensor &out,
                         const torch::Tensor &qkv_w, const torch::Tensor &qkv_b,
                         const torch::Tensor &op_w, const torch::Tensor &op_b,
                         const torch::Tensor &fi_w, const torch::Tensor &fi_b,
                         const torch::Tensor &fo_w, const torch::Tensor &fo_b,
                         const torch::Tensor &fn_w, const torch::Tensor &fn_b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mega_causal_forward", &mega_causal_forward,
        py::arg("x"), py::arg("out"), py::arg("qkv_w"), py::arg("qkv_b"),
        py::arg("op_w"), py::arg("op_b"), py::arg("fi_w"), py::arg("fi_b"),
        py::arg("fo_w"), py::arg("fo_b"), py::arg("fn_w"), py::arg("fn_b"));
}
