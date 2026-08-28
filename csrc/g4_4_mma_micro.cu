// G4.4 Stage 0a -- smallest possible mma.sync / ldmatrix micro-unit-test.
// Tiling / smem / register documentation for the GEMM this unblocks lives in
// csrc/g4_4_mma_gemm.cu; this file only pins the fragment addressing it relies on.
//
// PURPOSE: this repo contains no .cu file, no mma.sync and no ldmatrix
// anywhere (nor does the installed torch's headers), so there is no local
// example to copy exact fragment addressing from. Before building any tiled
// GEMM we verify, in isolation, that:
//   (1) ldmatrix.sync.aligned.m8n8.x4.shared.b16 loads the A fragment of
//       mma.sync.aligned.m16n8k16 in exactly the register/lane layout the
//       PTX ISA specifies, and
//   (2) ldmatrix.sync.aligned.m8n8.x2.shared.b16 loads the B fragment, and
//   (3) mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 (FP16 ACCUMULATE --
//       the whole point of G4.4; cuBLASLt can never emit this) computes
//       D = A*B + C with that layout.
//
// One warp, one 16x16x16 problem. Two independent outputs are produced from
// the SAME mma instruction operands built two different ways:
//   D_ld  -- fragments loaded with ldmatrix
//   D_man -- fragments assembled by hand from shared memory with the lane
//            mapping written out explicitly from the ISA table
// If D_ld == D_man == fp64 reference, both the ldmatrix addressing and the
// mma fragment layout are confirmed. If they disagree, the disagreement
// isolates WHICH of the two is wrong, which a single output could not.
//
// ISA fragment layout used (PTX ISA 8.x, mma.m16n8k16, .f16 type):
//   gid = lane >> 2   (0..7)      tig = lane & 3   (0..3)
//   A (16x16, .row), 4 x b32, 2 halves each:
//     a0 -> (row gid,   col 2*tig + {0,1})
//     a1 -> (row gid+8, col 2*tig + {0,1})
//     a2 -> (row gid,   col 2*tig + {0,1} + 8)
//     a3 -> (row gid+8, col 2*tig + {0,1} + 8)
//   B (16x8, .col), 2 x b32:
//     b0 -> (row 2*tig + {0,1},     col gid)
//     b1 -> (row 2*tig + {0,1} + 8, col gid)
//   C/D (16x8), 2 x b32 when accumulating in f16:
//     d0 -> (row gid,   col 2*tig + {0,1})
//     d1 -> (row gid+8, col 2*tig + {0,1})
//
// B is supplied as W [N=8, K=16] row-major, because the real GEMM this leads
// to is F.linear: Out[M,N] = In[M,K] @ W[N,K]^T, i.e. B = W^T and B's column n
// is W's row n. That makes W's rows contiguous in the k direction, which is
// exactly what the b0/b1 pairing wants.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t pack2(__half lo, __half hi) {
  uint32_t r;
  __half2 h = __halves2half2(lo, hi);
  memcpy(&r, &h, 4);
  return r;
}

}  // namespace

// A: [16,16] row-major fp16.  W: [8,16] row-major fp16 (B = W^T, [16,8]).
// D_ld, D_man: [16,8] row-major fp16.
__global__ void mma_micro_kernel(const __half *__restrict__ A,
                                 const __half *__restrict__ W,
                                 __half *__restrict__ D_ld,
                                 __half *__restrict__ D_man) {
  __shared__ __half sA[16 * 16];
  __shared__ __half sW[8 * 16];

  const int lane = threadIdx.x;  // single warp
  for (int i = lane; i < 256; i += 32) sA[i] = A[i];
  for (int i = lane; i < 128; i += 32) sW[i] = W[i];
  __syncwarp();

  const int gid = lane >> 2;
  const int tig = lane & 3;

  // ---------------- path 1: ldmatrix ----------------
  uint32_t a[4], b[2];
  {
    // x4: threads 0-7 give matrix0 rows, 8-15 matrix1, 16-23 matrix2,
    // 24-31 matrix3.  Register i receives matrix i.  For the m16n8k16 A
    // fragment the four 8x8 quadrants must land as
    //   reg0=(r0-7,c0-7) reg1=(r8-15,c0-7) reg2=(r0-7,c8-15) reg3=(r8-15,c8-15)
    // which is exactly:  row = lane % 16,  col = (lane / 16) * 8.
    const int arow = lane & 15;
    const int acol = (lane >> 4) * 8;
    uint32_t aaddr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&sA[arow * 16 + acol]));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(aaddr));

    // x2 on W [8,16]: matrix0 = W[n 0-7][k 0-7] -> b0, matrix1 = W[n 0-7][k
    // 8-15] -> b1.  Threads 0-7 address matrix0's rows, 8-15 matrix1's rows.
    const int wrow = lane & 7;
    const int wcol = ((lane >> 3) & 1) * 8;
    uint32_t waddr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&sW[wrow * 16 + wcol]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1])
                 : "r"(waddr));
  }

  // ---------------- path 2: hand-assembled ----------------
  uint32_t am[4], bm[2];
  {
    am[0] = pack2(sA[gid * 16 + 2 * tig], sA[gid * 16 + 2 * tig + 1]);
    am[1] = pack2(sA[(gid + 8) * 16 + 2 * tig], sA[(gid + 8) * 16 + 2 * tig + 1]);
    am[2] = pack2(sA[gid * 16 + 2 * tig + 8], sA[gid * 16 + 2 * tig + 9]);
    am[3] =
        pack2(sA[(gid + 8) * 16 + 2 * tig + 8], sA[(gid + 8) * 16 + 2 * tig + 9]);
    // b0 = B[k=2*tig+{0,1}][n=gid] = W[gid][2*tig+{0,1}]
    bm[0] = pack2(sW[gid * 16 + 2 * tig], sW[gid * 16 + 2 * tig + 1]);
    bm[1] = pack2(sW[gid * 16 + 2 * tig + 8], sW[gid * 16 + 2 * tig + 9]);
  }

  uint32_t c0 = 0, c1 = 0;
  uint32_t d[2], dm[2];
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(c0), "r"(c1));
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(dm[0]), "=r"(dm[1])
      : "r"(am[0]), "r"(am[1]), "r"(am[2]), "r"(am[3]), "r"(bm[0]), "r"(bm[1]),
        "r"(c0), "r"(c1));

  // D layout: d0 -> (gid, 2*tig+{0,1}), d1 -> (gid+8, 2*tig+{0,1})
  __half2 h0, h1, m0, m1;
  memcpy(&h0, &d[0], 4);
  memcpy(&h1, &d[1], 4);
  memcpy(&m0, &dm[0], 4);
  memcpy(&m1, &dm[1], 4);

  D_ld[gid * 8 + 2 * tig] = __low2half(h0);
  D_ld[gid * 8 + 2 * tig + 1] = __high2half(h0);
  D_ld[(gid + 8) * 8 + 2 * tig] = __low2half(h1);
  D_ld[(gid + 8) * 8 + 2 * tig + 1] = __high2half(h1);

  D_man[gid * 8 + 2 * tig] = __low2half(m0);
  D_man[gid * 8 + 2 * tig + 1] = __high2half(m0);
  D_man[(gid + 8) * 8 + 2 * tig] = __low2half(m1);
  D_man[(gid + 8) * 8 + 2 * tig + 1] = __high2half(m1);
}

void mma_micro_launch(const void *A, const void *W, void *D_ld, void *D_man,
                      cudaStream_t s) {
  mma_micro_kernel<<<1, 32, 0, s>>>(
      reinterpret_cast<const __half *>(A), reinterpret_cast<const __half *>(W),
      reinterpret_cast<__half *>(D_ld), reinterpret_cast<__half *>(D_man));
}
