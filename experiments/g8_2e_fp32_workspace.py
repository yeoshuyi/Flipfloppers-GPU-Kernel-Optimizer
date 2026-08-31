"""
G8.2 Phase 1c -- would an FP32 workspace rescue the split-K rebuild?

CUTLASS's stock GemmSplitKParallel CANNOT express the idea: at
.cutlass/include/cutlass/gemm/device/gemm_splitk_parallel.h:286 the workspace is
`TensorRef<ElementAccumulator_, ...>` -- the SAME template parameter the
mainloop mma accumulates in. So ElementAccumulator=half_t gives the 330 TF tier
AND an fp16 workspace; ElementAccumulator=float gives an fp32 workspace but
drops the mma back to the 165 TF tier. There is no stock combination of the two.

Rather than write a custom epilogue, measure the UPPER BOUND the whole direction
could ever reach: slice K by hand, FP16-accumulate each slice with the existing
CUTLASS FP16-accum config, and combine the partials EXACTLY in FP32. No
implementation can beat this -- an FP32 workspace at best avoids rounding
between slices, which is precisely what an exact fp32 sum does here.

If even this upper bound misses the 0.00180 ship ceiling, the direction is
closed regardless of how the reduction is implemented.

NOTE this is an ACCURACY bound only. N separate GEMM launches plus N FP32 adds
over the full [M,N] output is unambiguously slower than one fused GEMM, and
job 237 already showed the speed margin is gone by 4 slices.
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = "/work"
sys.path.insert(0, ROOT)
import torch_transformer_benchmark as B  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
ATOL, RTOL = 2e-3, 2e-2
SHIP_CEILING = 0.00180
CFG_F16 = 2          # accF16 TB128x128x32, the FP16-accumulate mainloop
MIN_SLICE_K = 32     # keep each slice a sane GEMM

CUTLASS = os.path.join(ROOT, ".cutlass")
SRC_CPP = os.path.join(ROOT, "csrc", "g4_6_cutlass_gemm.cpp")
SRC_CU = [os.path.join(ROOT, "csrc", f"g4_6_cutlass_cfg{i:02d}.cu")
          for i in range(28)]

OFFICIAL = [
    ("row01", 64, 128, 128, 4, 128, 4, 0.0013676),
    ("row08", 64, 128, 1024, 4, 1024, 4, 0.00141025),
    ("row09", 64, 128, 128, 1, 128, 4, 0.00145066),
    ("row11", 64, 128, 128, 16, 128, 4, 0.0013676),
]
SLICES = [1, 2, 4, 8, 16, 32]


def build_ext():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    return load(name="g8_2c_cutlass", sources=[SRC_CPP] + SRC_CU,
                build_directory=bd, with_cuda=True,
                extra_include_paths=[os.path.join(CUTLASS, "include"),
                                     os.path.join(CUTLASS, "tools", "util", "include")],
                extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                                   "--expt-relaxed-constexpr",
                                   "--expt-extended-lambda",
                                   "-diag-suppress", "179,177,20012"],
                verbose=False)


def build_models(bs, sl, dm, nh, ff, nl):
    cfg = B.TransformerConfig(batch_size=bs, seq_len=sl, d_model=dm,
                              num_heads=nh, ffn_dim=ff, num_layers=nl,
                              causal=True)
    cfg.validate()
    torch.manual_seed(0)
    base = B.BaselineTransformer(cfg).to(DEV, torch.float32).eval()
    opt = B.UserOptimizedTransformer(cfg).to(DEV, torch.float32).eval()
    B.copy_model_weights(base, opt, strict=True)
    return cfg, base, opt


def make_input(bs, sl, dm, seed=1234):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(bs, sl, dm, generator=g, device=DEV, dtype=torch.float32)


class Patch:
    """FP16-accumulate per K-slice, EXACT FP32 combine across slices."""

    def __init__(self, ext, nslices):
        self.ext, self.ns = ext, nslices

    def __enter__(self):
        self.orig = F.linear
        ext, ns, outer = self.ext, self.ns, self

        def patched(inp, weight, bias=None):
            if not (inp.dtype is torch.float16 and weight.dtype is torch.float16
                    and inp.is_cuda and inp.dim() in (2, 3)):
                return outer.orig(inp, weight, bias)
            M = inp.numel() // inp.shape[-1]
            K, N = inp.shape[-1], weight.shape[0]
            if K % ns or (K // ns) < MIN_SLICE_K:
                return outer.orig(inp, weight, bias)
            try:
                flat = inp.reshape(M, K)
                step = K // ns
                zb = torch.zeros(N, device=inp.device, dtype=inp.dtype)
                acc = torch.zeros(M, N, device=inp.device, dtype=torch.float32)
                for i in range(ns):
                    a = flat[:, i * step:(i + 1) * step].contiguous()
                    w = weight[:, i * step:(i + 1) * step].contiguous()
                    o = torch.empty(M, N, device=inp.device, dtype=inp.dtype)
                    nb = ext.cfg_workspace_bytes(CFG_F16, M, N, step)
                    ws = torch.empty(max(int(nb), 1), device=inp.device,
                                     dtype=torch.uint8)
                    ext.cutlass_gemm(CFG_F16, a, w, zb, o, ws)
                    acc += o.float()            # exact FP32 combine
                if bias is not None:
                    acc += bias.float()
                return acc.to(inp.dtype).reshape(*inp.shape[:-1], N)
            except Exception:
                return outer.orig(inp, weight, bias)

        F.linear = patched
        B.F.linear = patched
        return self

    def __exit__(self, *a):
        F.linear = self.orig
        B.F.linear = self.orig
        return False


@torch.no_grad()
def main():
    ext = build_ext()
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    print(f"mainloop: cfg[{CFG_F16}] = {ext.cfg_name(CFG_F16)}")
    print(f"ship ceiling {SHIP_CEILING};  budget abs<{ATOL} OR rel<{RTOL}\n")
    hdr = f"{'shape':>6} {'K':>5} {'shipped':>10}" + \
          "".join(f"{str(n) + ' slice':>16}" for n in SLICES)
    print(hdr); print("-" * len(hdr))
    ship = {}
    for lab, bs, sl, dm, nh, ff, nl, shipped in OFFICIAL:
        _, base, opt = build_models(bs, sl, dm, nh, ff, nl)
        x = make_input(bs, sl, dm)
        mask = torch.ones(bs, sl, dtype=torch.bool, device=DEV)
        ref = base(x, mask)
        opt._compiled_causal = opt._optimized_forward_causal
        cells = []
        for ns in SLICES:
            if dm % ns or (dm // ns) < MIN_SLICE_K:
                cells.append(f"{'slice<32':>16}"); continue
            try:
                with Patch(ext, ns):
                    y = opt(x, mask)
                r = B.compare_outputs(ref, y, RTOL, ATOL)
                ok = r.failed_elements == 0
                sh = ok and r.max_abs_error <= SHIP_CEILING
                cells.append(f"{r.max_abs_error:>10.3e} "
                             f"{('SHIP' if sh else ('rel' if ok else 'FAIL')):>5}")
                if sh:
                    ship.setdefault(lab, []).append(ns)
                del y
            except Exception as e:
                cells.append(f"{('x:' + type(e).__name__)[:15]:>16}")
            torch.cuda.empty_cache()
        print(f"{lab:>6} {dm:>5} {shipped:>10.5f}" + "".join(cells))
        del base, opt, x, mask, ref
        torch.cuda.empty_cache()
    print(f"\n  shippable with an EXACT FP32 combine: {ship if ship else 'NONE'}")
    if not ship:
        print("\n  This is the UPPER BOUND for the whole parallel-split-K direction:\n"
              "  the combine is exact, so no FP32 workspace can do better. The loss\n"
              "  is inside each slice -- the mma's .f16 accumulator rounds the\n"
              "  partial before any reduction ever sees it -- which is why the hand\n"
              "  kernel's WITHIN-loop FP32 carry beats every across-slice scheme.")
    print("\nG8_2E_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
