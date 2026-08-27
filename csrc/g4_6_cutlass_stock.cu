// G4.6 Phase 0 -- toolchain feasibility gate for CUTLASS on this container.
//
// The type/tile/arch/epilogue block below is copied VERBATIM from the vendored
// clone's own example
//     .cutlass/examples/12_gemm_bias_relu/gemm_bias_relu.cu  (CUTLASS v4.7.1)
// which is this clone's closest current match to "TensorOp FP16 GEMM through
// cutlass::gemm::device::Gemm with a real (bias) epilogue".  NOTE: the guess
// in the plan -- examples/18_ampere_fp16_tensorop_gemm -- does NOT exist in
// v4.7.1; example 18 there is `18_ampere_fp64_tensorop_affine2_gemm`.  The two
// live candidates were checked and example 12 was chosen:
//     12_gemm_bias_relu          fp16 in, device::Gemm, LinearCombinationRelu
//                                bias epilogue, Sm75 tag   <- CHOSEN
//     14_ampere_tf32_tensorop_gemm  Sm80 tag + LinearCombination, but tf32 in
//
// Nothing here is a CUTLASS-internal edit: only the harness around the config
// (a plain pointer launcher instead of the example's HostTensor main()) is
// ours, so Gate 0a's "zero edits to CUTLASS internals" holds.
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/version.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"

// ---- verbatim from examples/12_gemm_bias_relu/gemm_bias_relu.cu ------------
using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = float;

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm75;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, ElementComputeEpilogue,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
    LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
    ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
// ---- end verbatim block ---------------------------------------------------

// Launcher over raw device pointers.  Layouts are the example's:
//   A column-major M x K (lda = M), B column-major K x N (ldb = K),
//   C = bias vector, column-major M x 1 with ldc = 0 (projects away N),
//   D column-major M x N (ldd = M).
// Returns 0 on success, or -(int)cutlass::Status on failure.
int cutlass_stock_launch(const void *A, const void *B, const void *bias,
                         void *D, int M, int N, int K, void *workspace,
                         size_t workspace_bytes, cudaStream_t stream,
                         size_t *workspace_needed) {
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA const, LayoutInputA> ref_a(
      static_cast<ElementInputA const *>(A), LayoutInputA(M));
  cutlass::TensorRef<ElementInputB const, LayoutInputB> ref_b(
      static_cast<ElementInputB const *>(B), LayoutInputB(K));
  cutlass::TensorRef<ElementOutput const, LayoutOutput> ref_c(
      static_cast<ElementOutput const *>(bias), LayoutOutput(0));
  cutlass::TensorRef<ElementOutput, LayoutOutput> ref_d(
      static_cast<ElementOutput *>(D), LayoutOutput(M));

  typename Gemm::Arguments args{problem_size, ref_a,  ref_b,
                                ref_c,        ref_d,  {ElementComputeEpilogue(1)},
                                1};

  size_t need = Gemm::get_workspace_size(args);
  if (workspace_needed) *workspace_needed = need;
  // Query-only mode: A == nullptr means "just tell me the workspace size".
  // Without this the query would fall through and launch the kernel on null
  // pointers (an illegal access).
  if (A == nullptr) return 0;
  if (need > workspace_bytes) return -1000;

  Gemm op;
  cutlass::Status st = op.can_implement(args);
  if (st != cutlass::Status::kSuccess) return -(int)st;
  st = op.initialize(args, workspace, stream);
  if (st != cutlass::Status::kSuccess) return -(int)st;
  st = op(stream);
  if (st != cutlass::Status::kSuccess) return -(int)st;
  return 0;
}

int cutlass_version_major() { return CUTLASS_MAJOR; }
int cutlass_version_minor() { return CUTLASS_MINOR; }
int cutlass_version_patch() { return CUTLASS_PATCH; }
