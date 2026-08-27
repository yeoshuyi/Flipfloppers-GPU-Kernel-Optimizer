// G4.5 -- standalone single-kernel translation unit for SASS-level work on
// G4.4's winning configuration.
//
// docs/PROGRESS.md step 37 measured cfg[11] (accF16 BM128 BN128 BK64 stg2
// split0) as the fastest FP16-accumulate instantiation at the primary shape
// (qkv / large_batch, M=32768 K=512 N=1536): 283.29 us = 181.93 TFLOPS =
// 55.1% of the 330.3 TF FP16-accumulate tier ceiling, against cuBLASLt at
// 91.2% of its own (2x lower) tier.
//
// This file exists ONLY so that `nvcc -cubin` emits a cubin containing
// exactly ONE kernel -- the one being studied -- rather than all 26 template
// instantiations that csrc/g4_4_mma_gemm.cu's host dispatch pulls in. A
// single-kernel cubin is what an external SASS assembler can round-trip
// without dragging in unrelated code, and it makes any instruction-level edit
// unambiguously attributable.
//
// The kernel is INCLUDED and EXPLICITLY INSTANTIATED, never copied. There is
// still exactly one definition of mma_gemm_kernel<> in this repo, and the
// explicit instantiation below produces bit-for-bit the same kernel the torch
// extension compiles for cfg[11] -- no wrapper, no extra __device__ hop, so
// no chance of the SASS under study differing from the SASS that was
// measured. G4_4_NO_HOST_DISPATCH suppresses only the host-side launcher
// (the thing that would instantiate the other 25 configs); the __global__
// template itself is untouched.

#define G4_4_NO_HOST_DISPATCH 1
#include "g4_4_mma_gemm.cu"

// cfg[11] == kCfg[11] in csrc/g4_4_mma_gemm.cpp == {BM,BN,BK,NSTAGE,SPLIT,ACCF32}
//         == {128, 128, 64, 2, 0, 0}.  Confirmed against the CFG(11, ...) line
//         in g4_4_mma_gemm.cu's dispatch switch -- both agree, so the index
//         numbering has NOT shifted since step 37.
//
// Shared memory this instantiation needs:
//   NSTAGE * (BM*BK + BN*BK) * sizeof(__half)
//   = 2 * (128*64 + 128*64) * 2 = 65536 B = 64 KB
// which is above the 48 KB default, so any host loader MUST opt in via
// cudaFuncAttributeMaxDynamicSharedMemorySize / CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_
// SHARED_SIZE_BYTES before launching.
template __global__ void mma_gemm_kernel<128, 128, 64, 2, 0, 0>(
    const __half *__restrict__, const __half *__restrict__,
    const __half *__restrict__, __half *__restrict__, int, int, int);
