"""
G8.2 Phase 0 -- does FP16-accumulate + an FP32 carry fit the accuracy budget at
the OFFICIAL matrix's K?

The proposal: run the tensor core at the un-throttled FP16-accumulate rate
(330.3 TF vs 165.2 TF -- NVIDIA's own Ada whitepapers confirm GeForce halves the
FP32-accumulate path while the same AD102 die does not on RTX 6000 Ada), then
rebuild FP32 precision OUTSIDE the tensor core to recover accuracy.

That mechanism already exists: `SPLIT` in csrc/g4_4_mma_gemm.cu accumulates in
FP16 within a chunk of SPLIT columns of K and promotes to an FP32 carry at each
chunk boundary. It was measured once, in job 129, and closed -- but at K=512,
the project's internal d512 shapes. The OFFICIAL matrix runs these GEMMs at
K = d_model, which is 128 on ten of fourteen rows. FP16 accumulation error grows
with K, so the majority of the matrix has never been tested, at a quarter of the
K that failed.

This phase answers ONLY the accuracy question, with no new CUDA, because it can
kill the whole idea for ~10 minutes of GPU. Kill criterion: if no row clears the
budget with margin at SPLIT=64, stop -- a chunk below BK=64 IS FP32 accumulation
(csrc comment), so the ladder has no rungs left.

Method: monkeypatch F.linear so the three fp16 causal GEMMs (qkv, out_proj,
ffn_in) route through the mma kernel, score end to end with the harness's own
compare_outputs against the frozen BaselineTransformer. `ffn_out` is excluded by
construction -- it runs fp32 and is load-bearing. Nothing shipped is touched.

Run via infra/slurm/g8_2_fp16accum_carry.sbatch. sbatch only.
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
torch.backends.cudnn.allow_tf32 = True
ATOL, RTOL = 2e-3, 2e-2
SHIP_CEILING = 0.00180   # docs/ACCURACY_BUDGET.md hard ship ceiling (90%)

SRC_CU = os.path.join(ROOT, "csrc", "g4_4_mma_gemm.cu")
SRC_CPP = os.path.join(ROOT, "csrc", "g4_4_mma_gemm.cpp")

# cfg index -> (BM, BN, BK, NSTAGE, SPLIT, ACCF32), from csrc/g4_4_mma_gemm.cpp
CFG = {
    "accF32 (reference tier)": 15,   # 64,128,64,3,   0, 1
    "accF16 no carry":          0,   # 64,128,64,3,   0, 0
    "accF16 SPLIT 256":         5,   # 64,128,64,3, 256, 0
    "accF16 SPLIT 128":         7,   # 64,128,64,3, 128, 0
    "accF16 SPLIT 64":          9,   # 64,128,64,3,  64, 0
    "accF16 BK32 SPLIT 64":    26,   # 64,128,32,7,  64, 0
    "accF16 BK32 SPLIT 32":    27,   # 64,128,32,7,  32, 0  <- finest non-FP32
}
BM, BN, BK = 64, 128, 32   # BK=32 so the finer-carry cfgs are expressible

# label, B, S, d, H, ffn, layers, current shipped max_abs (run 216)
OFFICIAL = [
    ("row01", 64, 128, 128, 4, 128, 4, 0.0013676),
    ("row02", 1, 128, 128, 4, 128, 4, 0.0013676),
    ("row03", 4, 128, 128, 4, 128, 4, 0.0013676),
    ("row04", 16, 128, 128, 4, 128, 4, 0.0013676),
    ("row05", 128, 128, 128, 4, 128, 4, 0.0013676),
    ("row06", 10000, 128, 128, 4, 128, 4, 0.00195017),
    ("row07", 64, 128, 32, 4, 32, 4, 0.00211424),
    ("row08", 64, 128, 1024, 4, 1024, 4, 0.00141025),
    ("row09", 64, 128, 128, 1, 128, 4, 0.00145066),
    ("row10", 64, 128, 128, 2, 128, 4, 0.00138116),
    ("row11", 64, 128, 128, 16, 128, 4, 0.0013676),
    ("row12", 64, 32, 128, 4, 128, 4, 0.00140575),
    ("row13", 64, 1024, 128, 4, 128, 4, 0.0013676),
]


def build_ext():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    os.makedirs(bd, exist_ok=True)
    return load(name="g8_2_mma_gemm", sources=[SRC_CPP, SRC_CU],
                build_directory=bd, with_cuda=True,
                extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                                   "-diag-suppress", "179"],
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


def expressible(dm, ffn):
    """Can the existing tiling run this shape's three fp16 GEMMs?
    Kernel requires M%BM==0, N%BN==0, K%BK==0 (and K%SPLIT==0)."""
    Ks = {dm}
    Ns = {3 * dm, dm, ffn}
    bad = [f"K={k}" for k in Ks if k % BK] + [f"N={n}" for n in Ns if n % BN]
    return (not bad), bad


class Patch:
    """Route the fp16 causal GEMMs through the mma kernel; leave the rest alone.

    fp32 tensors fall through untouched, which excludes `ffn_out` (fp32 weights,
    load-bearing for this path's accuracy) by construction rather than by name.
    """

    def __init__(self, ext, cfg):
        self.ext, self.cfg, self.n, self.skipped = ext, cfg, 0, 0

    def __enter__(self):
        self.orig = F.linear
        ext, cfg, outer = self.ext, self.cfg, self

        def patched(inp, weight, bias=None):
            if (inp.dtype is torch.float16 and weight.dtype is torch.float16
                    and inp.is_cuda and inp.dim() in (2, 3)):
                M = inp.numel() // inp.shape[-1]
                K, N = inp.shape[-1], weight.shape[0]
                if M % BM == 0 and N % BN == 0 and K % BK == 0:
                    flat = inp.reshape(M, K).contiguous()
                    w = weight.contiguous()
                    b = bias.contiguous() if bias is not None else None
                    out = ext.mma_linear(cfg, flat, w, b)
                    outer.n += 1
                    return out.reshape(*inp.shape[:-1], N)
                outer.skipped += 1
            return outer.orig(inp, weight, bias)

        F.linear = patched
        B.F.linear = patched
        return self

    def __exit__(self, *a):
        F.linear = self.orig
        B.F.linear = self.orig
        return False


@torch.no_grad()
def run():
    ext = build_ext()
    print(f"extension built: {ext.num_cfg()} configs")
    for name, c in CFG.items():
        print(f"  cfg[{c}] = {ext.cfg_name(c)}   ({name})")

    print(f"\nbudget: abs < {ATOL} OR rel < {RTOL}\n")
    hdr = f"{'shape':>6} {'d':>5} {'K':>5} {'shipped':>10} " + \
          "".join(f"{k.replace('accF16 ',''):>18}" for k in CFG)
    print(hdr)
    print("-" * len(hdr))
    survivors = {}
    for lab, bs, sl, dm, nh, ff, nl, shipped in OFFICIAL:
        ok, why = expressible(dm, ff)
        if not ok:
            print(f"{lab:>6} {dm:>5} {dm:>5} {shipped:>10.5f}   "
                  f"NOT EXPRESSIBLE with the existing tiling ({', '.join(why)}; "
                  f"BM/BN/BK = {BM}/{BN}/{BK})")
            continue
        _, base, opt = build_models(bs, sl, dm, nh, ff, nl)
        x = make_input(bs, sl, dm)
        mask = torch.ones(bs, sl, dtype=torch.bool, device=DEV)
        ref = base(x, mask)
        # eager causal path so the monkeypatch is visible (torch.compile would
        # trace through it); forward() still builds the folded weights first
        opt._compiled_causal = opt._optimized_forward_causal
        cells, best = [], None
        for name, c in CFG.items():
            if c in (5, 7, 9, 26, 27):
                split = {5: 256, 7: 128, 9: 64, 26: 64, 27: 32}[c]
                if dm % split:
                    cells.append(f"{'K%SPLIT!=0':>22}")
                    continue
            try:
                with Patch(ext, c) as p:
                    y = opt(x, mask)
                r = B.compare_outputs(ref, y, RTOL, ATOL)
                # Two bars. `failed==0` is the HARNESS gate (disjunctive, so an
                # element over atol can still pass on the relative arm).
                # SHIP_CEILING is docs/ACCURACY_BUDGET.md's hard line, which is
                # what actually decides whether a lossy change is shippable.
                harness = r.failed_elements == 0
                shippable = harness and r.max_abs_error <= SHIP_CEILING
                tag = "SHIP" if shippable else ("rel-only" if harness else "FAIL")
                cells.append(f"{r.max_abs_error:>12.3e} {tag:>9}")
                if shippable and "accF16" in name:
                    best = name
                del y
            except Exception as e:
                cells.append(f"{('ERR:' + type(e).__name__):>22}")
            torch.cuda.empty_cache()
        print(f"{lab:>6} {dm:>5} {dm:>5} {shipped:>10.5f}" + "".join(cells))
        if best:
            survivors[lab] = best
        del base, opt, x, mask, ref
        torch.cuda.empty_cache()

    print(f"\nrows where an FP16-accumulate config is SHIPPABLE (max_abs <= 0.00180): "
          f"{survivors if survivors else 'NONE'}")
    if not survivors:
        print("\nKILL CRITERION MET: no official row clears the budget with an\n"
              "FP16-accumulate config. A chunk below BK=64 IS FP32 accumulation\n"
              "(csrc/g4_4_mma_gemm.cu), so the ladder has no rungs left and no\n"
              "amount of kernel engineering (CUTLASS or otherwise) can recover\n"
              "it. Phases 1-3 are not justified.")
    else:
        print("\nPhase 0 leaves live shapes -> proceed to Phase 1 "
              "(CUTLASS + split-K) for those K/M only.")
    print("\nG8_2_PHASE0_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
