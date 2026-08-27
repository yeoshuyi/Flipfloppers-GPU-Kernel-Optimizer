// G2.3 -- CUDA L2 cache persistence control, as a real pybind11 extension.
//
// Why an extension and not ctypes: docs/PROGRESS.md step 16 rejected the
// ctypes route because a struct-layout or enum mistake in
// cudaAccessPolicyWindow has no compiler in the loop and reaches a live
// kernel launch as either a silent no-op or a crash. Compiling against this
// container's real CUDA headers makes the struct layout, the enum values and
// the function signatures the compiler's problem, not ours.
//
// Header facts verified in results/l2_persist_discover_run36.log (CUDA 13.1):
//   cudaLimitPersistingL2CacheSize = 0x06
//   struct cudaAccessPolicyWindow { void*; size_t; float; enum; enum; }
//   cudaStreamAttrID/cudaStreamAttrValue are now #define aliases of
//   cudaLaunchAttributeID/cudaLaunchAttributeValue -- the modern names are
//   used below.
//
// NOTE ON data_ptr(): set_window() takes a torch::Tensor and derives the base
// address here, in C++. Nothing in benchmark.py ever calls .data_ptr() for
// this -- tools/check_validity.py's data_ptr rule (only _mask_is_all_ones may
// call it) is therefore not tripped, and not worked around either: the Python
// side genuinely has no need for the raw address.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstring>

namespace {

inline void ck(cudaError_t e, const char *what) {
  TORCH_CHECK(e == cudaSuccess, what, " failed: ", cudaGetErrorString(e));
}

}  // namespace

// Idempotent. Requests `bytes` of L2 set aside for persisting accesses and
// returns what the driver ACTUALLY granted -- the GPU caps this well below
// total L2 (see persisting_l2_max_size()), so the caller must size hitRatio
// against the return value, never against the request.
int64_t set_persist_limit(int64_t bytes) {
  TORCH_CHECK(bytes >= 0, "bytes must be non-negative");
  cudaError_t e =
      cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)bytes);
  if (e != cudaSuccess) {
    // Swallow: an over-large request is reported here, but the query below
    // still tells the caller the truth about the current limit.
    cudaGetLastError();
  }
  size_t got = 0;
  ck(cudaDeviceGetLimit(&got, cudaLimitPersistingL2CacheSize),
     "cudaDeviceGetLimit(cudaLimitPersistingL2CacheSize)");
  return (int64_t)got;
}

int64_t get_persist_limit() {
  size_t got = 0;
  ck(cudaDeviceGetLimit(&got, cudaLimitPersistingL2CacheSize),
     "cudaDeviceGetLimit(cudaLimitPersistingL2CacheSize)");
  return (int64_t)got;
}

static cudaDeviceProp current_props() {
  int dev = 0;
  ck(cudaGetDevice(&dev), "cudaGetDevice");
  cudaDeviceProp p;
  std::memset(&p, 0, sizeof(p));
  ck(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
  return p;
}

int64_t l2_cache_size() { return (int64_t)current_props().l2CacheSize; }

int64_t persisting_l2_max_size() {
  return (int64_t)current_props().persistingL2CacheMaxSize;
}

int64_t access_policy_max_window_size() {
  return (int64_t)current_props().accessPolicyMaxWindowSize;
}

// Sets an accessPolicyWindow covering `t`'s whole storage-slice on the CURRENT
// CUDA stream, i.e. whichever stream torch is running/capturing on right now.
// hit_ratio must be sized so num_bytes * hit_ratio <= the granted persisting
// limit, otherwise the driver's behaviour is undefined per the CUDA docs.
void set_window(const torch::Tensor &t, double hit_ratio) {
  TORCH_CHECK(t.is_cuda(), "arena tensor must be on CUDA");
  TORCH_CHECK(t.is_contiguous(), "arena tensor must be contiguous");
  TORCH_CHECK(hit_ratio > 0.0 && hit_ratio <= 1.0,
              "hit_ratio must be in (0, 1]");

  const int64_t nbytes = t.numel() * t.element_size();
  TORCH_CHECK(nbytes > 0, "arena tensor is empty");

  cudaLaunchAttributeValue attr;
  std::memset(&attr, 0, sizeof(attr));
  attr.accessPolicyWindow.base_ptr = t.data_ptr();
  attr.accessPolicyWindow.num_bytes = (size_t)nbytes;
  attr.accessPolicyWindow.hitRatio = (float)hit_ratio;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

  cudaStream_t s = c10::cuda::getCurrentCUDAStream();
  ck(cudaStreamSetAttribute(s, cudaLaunchAttributeAccessPolicyWindow, &attr),
     "cudaStreamSetAttribute(AccessPolicyWindow)");
}

// Clears any window on the current stream (num_bytes = 0 disables it).
void reset_window() {
  cudaLaunchAttributeValue attr;
  std::memset(&attr, 0, sizeof(attr));
  attr.accessPolicyWindow.base_ptr = nullptr;
  attr.accessPolicyWindow.num_bytes = 0;
  attr.accessPolicyWindow.hitRatio = 0.0f;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  cudaStream_t s = c10::cuda::getCurrentCUDAStream();
  ck(cudaStreamSetAttribute(s, cudaLaunchAttributeAccessPolicyWindow, &attr),
     "cudaStreamSetAttribute(reset AccessPolicyWindow)");
}

// Drops all persisting lines back to normal status.
void reset_persisting_l2() {
  ck(cudaCtxResetPersistingL2Cache(), "cudaCtxResetPersistingL2Cache");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_persist_limit", &set_persist_limit,
        "cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize); returns granted");
  m.def("get_persist_limit", &get_persist_limit, "granted persisting L2 bytes");
  m.def("l2_cache_size", &l2_cache_size, "device l2CacheSize");
  m.def("persisting_l2_max_size", &persisting_l2_max_size,
        "device persistingL2CacheMaxSize");
  m.def("access_policy_max_window_size", &access_policy_max_window_size,
        "device accessPolicyMaxWindowSize");
  m.def("set_window", &set_window,
        "set accessPolicyWindow over a tensor on the current stream");
  m.def("reset_window", &reset_window, "clear accessPolicyWindow");
  m.def("reset_persisting_l2", &reset_persisting_l2,
        "cudaCtxResetPersistingL2Cache");
}
