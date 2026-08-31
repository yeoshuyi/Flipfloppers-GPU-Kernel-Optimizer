"""
G8.2 Phase 1b -- accuracy of the CUTLASS FP16-accumulate configs (with and
without split-K) on the official matrix.

Phase 1 (job 237) refuted the Phase-2 generalisation: at K=128 the accumulate
tier really is inert, but at K=1024 (official row 8) CUTLASS FP16-accumulate
hits 239.4 TF against cuBLAS's 154.3 -- 1.55x faster, and ABOVE the 165.2 TF
FP32-accum ceiling, so the tier is genuinely engaged. Its split-K variants also
beat cuBLAS (cfg27 1.40x, cfg24 1.29x).

That makes row 8 a live candidate worth ~18% of its forward, and the only thing
left to decide it is accuracy. Phase 0 measured the HAND kernel's within-block
carry; CUTLASS's split-K is a different reduction (serial across CTA slices,
through the fp16 output), so it has to be measured rather than assumed.

Scored against docs/ACCURACY_BUDGET.md's hard ship ceiling of 0.00180, not the
harness's disjunctive failed==0.
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

CUTLASS = os.path.join(ROOT, ".cutlass")
SRC_CPP = os.path.join(ROOT, "csrc", "g4_6_cutlass_gemm.cpp")
NUM_CFG = 28
SRC_CU = [os.path.join(ROOT, "csrc", f"g4_6_cutlass_cfg{i:02d}.cu")
          for i in range(NUM_CFG)]

# the configs that beat cuBLAS in job 237, plus the FP32-accum control
CFG = {"cfg12 accF32 (control)": 12, "cfg2 accF16 no split": 2,
       "cfg24 accF16 splitK2": 24, "cfg27 accF16 splitK2": 27,
       "cfg25 accF16 splitK4": 25}

OFFICIAL = [
    ("row01", 64, 128, 128, 4, 128, 4, 0.0013676),
    ("row06", 10000, 128, 128, 4, 128, 4, 0.00195017),
    ("row07", 64, 128, 32, 4, 32, 4, 0.00211424),
    ("row08", 64, 128, 1024, 4, 1024, 4, 0.00141025),
    ("row09", 64, 128, 128, 1, 128, 4, 0.00145066),
    ("row11", 64, 128, 128, 16, 128, 4, 0.0013676),
    ("row13", 64, 1024, 128, 4, 128, 4, 0.0013676),
]


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
    """Route the fp16 causal GEMMs through CUTLASS. fp32 falls through, which
    excludes the load-bearing fp32 `ffn_out` by construction."""

    def __init__(self, ext, cfg):
        self.ext, self.cfg = ext, cfg

    def __enter__(self):
        self.orig = F.linear
        ext, c, outer = self.ext, self.cfg, self

        def patched(inp, weight, bias=None):
            if (inp.dtype is torch.float16 and weight.dtype is torch.float16
                    and inp.is_cuda and inp.dim() in (2, 3)):
                M = inp.numel() // inp.shape[-1]
                K, N = inp.shape[-1], weight.shape[0]
                try:
                    flat = inp.reshape(M, K).contiguous()
                    w = weight.contiguous()
                    b = (bias.contiguous() if bias is not None
                         else torch.zeros(N, device=inp.device, dtype=inp.dtype))
                    out = torch.empty(M, N, device=inp.device, dtype=inp.dtype)
                    nb = ext.cfg_workspace_bytes(c, M, N, K)
                    ws = torch.empty(max(int(nb), 1), device=inp.device,
                                     dtype=torch.uint8)
                    ext.cutlass_gemm(c, flat, w, b, out, ws)
                    return out.reshape(*inp.shape[:-1], N)
                except Exception:
                    pass
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
    print(f"budget: abs < {ATOL} OR rel < {RTOL};  ship ceiling {SHIP_CEILING}\n")
    hdr = f"{'shape':>6} {'K':>5} {'shipped':>10}" + \
          "".join(f"{n.split()[0]:>20}" for n in CFG)
    print(hdr); print("-" * len(hdr))
    ship = {}
    for lab, bs, sl, dm, nh, ff, nl, shipped in OFFICIAL:
        _, base, opt = build_models(bs, sl, dm, nh, ff, nl)
        x = make_input(bs, sl, dm)
        mask = torch.ones(bs, sl, dtype=torch.bool, device=DEV)
        ref = base(x, mask)
        opt._compiled_causal = opt._optimized_forward_causal
        cells = []
        for name, c in CFG.items():
            try:
                with Patch(ext, c):
                    y = opt(x, mask)
                r = B.compare_outputs(ref, y, RTOL, ATOL)
                ok = r.failed_elements == 0
                sh = ok and r.max_abs_error <= SHIP_CEILING
                cells.append(f"{r.max_abs_error:>12.3e} "
                             f"{('SHIP' if sh else ('rel' if ok else 'FAIL')):>6}")
                if sh and "accF16" in name:
                    ship.setdefault(lab, []).append(name)
                del y
            except Exception as e:
                cells.append(f"{('x:' + type(e).__name__)[:19]:>20}")
            torch.cuda.empty_cache()
        print(f"{lab:>6} {dm:>5} {shipped:>10.5f}" + "".join(cells))
        del base, opt, x, mask, ref
        torch.cuda.empty_cache()
    print(f"\n  shippable (max_abs <= {SHIP_CEILING}): {ship if ship else 'NONE'}")
    print("\nG8_2D_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
