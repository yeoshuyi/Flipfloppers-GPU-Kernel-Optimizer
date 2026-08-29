"""
G7.0 -- verification for _chunked_forward_causal (benchmark.py).

That method is the eager, sequence-chunked causal forward that lets the shipped
UserOptimizedTransformer *execute* official Row 14
(B=32 d=1024 H=16 S=100000 L=2 ffn=1024, causal) -- and any larger causal shape
-- on one 24 GB RTX 4090. The frozen harness cannot score Row 14 at all (the
FP32 reference OOMs in generate_random_case / baseline's [B,H,S,S] scores
before our model is ever called, and run_accuracy_tests has no try/except), so
this probe is the standalone proof that the capability is real and correct.

Checks (a compact report, PASS/FAIL per check + OVERALL):

  1. sdpa_prefix_causal   -- F.scaled_dot_product_attention(q[c0:c1], k[:c1],
     v[:c1], is_causal=True) equals the causal-prefix slice of the full
     attention, for every memory-lean backend. This is the load-bearing
     assumption of the whole design -- FAIL here => sys.exit(1).
  2. gate                 -- _would_oom_causal is False for all 13 official
     rows (the gate never misfires on a scored shape); with the threshold
     lowered, forward()'s auto-route is bit-identical to a direct
     _chunked_forward_causal call.
  3. equivalence          -- small causal shape: fp16-store chunked vs the
     frozen BaselineTransformer and vs the shipped compiled path -> failed==0.
  4. oversize_capability  -- three synthetic >24 GB causal shapes (Row 14 plus
     two others) run end to end: finite output, peak VRAM, adaptive chunk_q,
     latency; plus the CHUNK_COMPILE on/off delta on Row 14.
  5. row14_accuracy       -- Row 14 dims at reduced batch: fp16-store chunked
     vs fp32-store chunked, within atol 0.002 / rtol 0.02 -> failed==0.

Run via infra/slurm/g7_0_chunked_oversize.sbatch (needs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for the tight Row-14 budget).
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/work")
import benchmark as B  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ATOL, RTOL = 2e-3, 2e-2
OFFICIAL_BSD = [  # (batch, seq_len, d_model) for the 13 runnable official rows
    (64, 128, 128), (1, 128, 128), (4, 128, 128), (16, 128, 128),
    (128, 128, 128), (10000, 128, 128), (64, 128, 32), (64, 128, 1024),
    (64, 128, 128), (64, 128, 128), (64, 128, 128), (64, 32, 128),
    (64, 1024, 128),
]


def gb(nbytes):
    return nbytes / 1024 ** 3


def make_input_fp16(bn, sn, dn, seed, tile=8192):
    """N(0,1) fp16 input built in sequence tiles -- never a 2x transient."""
    x = torch.empty(bn, sn, dn, dtype=torch.float16, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for c0 in range(0, sn, tile):
        x[:, c0:min(c0 + tile, sn)].normal_(0.0, 1.0, generator=g)
    return x


def build_models(bs, sl, dm, nh, ff, nl, dtype=torch.float32):
    cfg = B.TransformerConfig(batch_size=bs, seq_len=sl, d_model=dm,
                              num_heads=nh, ffn_dim=ff, num_layers=nl,
                              causal=True)
    cfg.validate()
    torch.manual_seed(0)
    base = B.BaselineTransformer(cfg).to(DEV, dtype).eval()
    opt = B.UserOptimizedTransformer(cfg).to(DEV, dtype).eval()
    B.copy_model_weights(base, opt, strict=True)
    return cfg, base, opt


def time_call(fn, warmup=1, iters=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts


# --------------------------------------------------------------------------
# 1. SDPA prefix-causal alignment
# --------------------------------------------------------------------------
@torch.no_grad()
def check_sdpa_prefix_causal():
    print("\n== 1. sdpa_prefix_causal ==", flush=True)
    bn, hn, sn, hd = 2, 3, 96, 32
    c0, c1 = 32, 64
    ok_any = False
    for dtype, backends, atol in (
        (torch.float32, [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION], 2e-4),
        (torch.float16, [SDPBackend.FLASH_ATTENTION,
                         SDPBackend.EFFICIENT_ATTENTION], 3e-3),
    ):
        g = torch.Generator(device=DEV).manual_seed(11)
        q = torch.randn(bn, hn, sn, hd, generator=g, device=DEV, dtype=dtype)
        k = torch.randn(bn, hn, sn, hd, generator=g, device=DEV, dtype=dtype)
        v = torch.randn(bn, hn, sn, hd, generator=g, device=DEV, dtype=dtype)
        for bk in backends:
            try:
                with sdpa_kernel([bk]):
                    full = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True, scale=1.0)
                    part = F.scaled_dot_product_attention(
                        q[:, :, c0:c1], k[:, :, :c1], v[:, :, :c1],
                        is_causal=True, scale=1.0)
            except (RuntimeError, ValueError) as e:
                print(f"  {str(dtype):>15} {bk.name:<20} unavailable "
                      f"({str(e)[:50]})")
                continue
            err = (part.float() - full[:, :, c0:c1].float()).abs().max().item()
            good = err <= atol
            ok_any |= good
            print(f"  {str(dtype):>15} {bk.name:<20} max|part-full[c0:c1]| "
                  f"= {err:.2e}   {'ok' if good else 'MISMATCH'}")
            if not good:
                print("  FAIL: SDPA(is_causal=True) is NOT bottom-right / "
                      "prefix-causal aligned on this backend -- the chunked "
                      "design is invalid. (fallback: explicit banded attn_mask.)")
                sys.exit(1)
    if not ok_any:
        print("  FAIL: no memory-lean SDPA backend available at all")
        sys.exit(1)
    print("  PASS")
    return True


# --------------------------------------------------------------------------
# 2. gate: never misfires on an official row; auto-route == direct call
# --------------------------------------------------------------------------
@torch.no_grad()
def check_gate():
    print("\n== 2. gate ==", flush=True)
    f = B.UserOptimizedTransformer._would_oom_causal
    worst = max(OFFICIAL_BSD, key=lambda r: r[0] * r[1] * r[2])
    for (b, s, d) in OFFICIAL_BSD:
        assert not f(b, s, d), f"gate misfired on official shape {(b, s, d)}"
    print(f"  _would_oom_causal False for all 13 official rows "
          f"(worst B*S*d = {worst[0] * worst[1] * worst[2]:,} @ {worst}, "
          f"threshold {B._CHUNK_ACT_ELEMS:,})")

    _, base, opt = build_models(2, 4096, 128, 4, 128, 2)
    x = make_input_fp16(2, 4096, 128, seed=1)
    mask = torch.ones(2, 4096, dtype=torch.bool, device=DEV)
    saved_elems, saved_q = B._CHUNK_ACT_ELEMS, B._CHUNK_Q
    try:
        B._CHUNK_ACT_ELEMS = 1_000_000          # 2*4096*128 = 1.05e6 -> trips
        B._CHUNK_Q = 1024                        # fixed => identical numerics
        assert f(2, 4096, 128), "lowered threshold did not trip the gate"
        y_auto = opt(x.clone(), mask)            # forward() -> auto-route
        y_dir = opt._chunked_forward_causal(x.clone(), mask)
    finally:
        B._CHUNK_ACT_ELEMS, B._CHUNK_Q = saved_elems, saved_q
    diff = (y_auto.float() - y_dir.float()).abs().max().item()
    good = diff <= 1e-5           # same implementation; <= flash atomic jitter
    print(f"  forward() auto-route vs direct _chunked_forward_causal: "
          f"max|.| = {diff:.2e}   {'PASS' if good else 'FAIL'}")
    del base, opt, x, y_auto, y_dir
    torch.cuda.empty_cache()
    return good


# --------------------------------------------------------------------------
# 3. equivalence: chunked == frozen baseline == shipped compiled path
# --------------------------------------------------------------------------
@torch.no_grad()
def check_equivalence():
    print("\n== 3. equivalence (B=2 S=4096 d=256 H=4 L=2 causal) ==", flush=True)
    _, base, opt = build_models(2, 4096, 256, 4, 256, 2)
    x = torch.randn(2, 4096, 256, device=DEV, dtype=torch.float32,
                    generator=torch.Generator(device=DEV).manual_seed(7))
    mask = torch.ones(2, 4096, dtype=torch.bool, device=DEV)

    ref = base(x, mask)                                       # frozen baseline
    saved_q = B._CHUNK_Q
    B._CHUNK_Q = 512
    try:
        chk16 = opt._chunked_forward_causal(x.clone().half(), mask)      # fp16
        chk32 = opt._chunked_forward_causal(x.clone().float(), mask,
                                            store=torch.float32)         # fp32
    finally:
        B._CHUNK_Q = saved_q
    shp = opt(x, mask)                                        # compiled shipped

    good = True
    for label, a, b in (("chunked(fp16) vs baseline", ref, chk16),
                        ("chunked(fp32) vs baseline", ref, chk32),
                        ("chunked(fp16) vs shipped  ", shp, chk16)):
        r = B.compare_outputs(a, b, RTOL, ATOL)
        p = r.failed_elements == 0
        good &= p
        print(f"  {label}: max_abs={r.max_abs_error:.3e} "
              f"max_rel={r.max_relative_error:.3e} "
              f"failed={r.failed_elements}/{r.total_elements}  "
              f"{'PASS' if p else 'FAIL'}")
    del base, opt, x, ref, chk16, chk32, shp
    torch.cuda.empty_cache()
    return good


# --------------------------------------------------------------------------
# 4. oversize capability: three >24 GB causal shapes, end to end
# --------------------------------------------------------------------------
@torch.no_grad()
def check_oversize_capability():
    print("\n== 4. oversize_capability ==", flush=True)
    shapes = [
        ("row14        ", 32, 100000, 1024, 16, 1024, 2),
        ("B8_S200k     ", 8, 200000, 1024, 16, 1024, 2),
        ("B64_S50k     ", 64, 50000, 1024, 16, 1024, 2),
    ]
    good = True
    for name, bs, sl, dm, nh, ff, nl in shapes:
        try:
            _, base, opt = build_models(bs, sl, dm, nh, ff, nl)
            del base
            x = make_input_fp16(bs, sl, dm, seed=1234)
            mask = torch.ones(bs, sl, dtype=torch.bool, device=DEV)
            torch.cuda.reset_peak_memory_stats()
            med, allts = time_call(lambda: opt(x, mask))
            peak = torch.cuda.max_memory_allocated()
            y = opt(x, mask)
            finite = bool(torch.isfinite(y).all())
            shape_ok = tuple(y.shape) == (bs, sl, dm)
            p = finite and shape_ok
            good &= p
            print(f"  {name} B*S*d={bs * sl * dm:>13,}  "
                  f"latency={med:7.1f} ms (all {['%.0f' % t for t in allts]})  "
                  f"peak={gb(peak):5.2f} GB  finite={finite} shape_ok={shape_ok} "
                  f" {'PASS' if p else 'FAIL'}")
            del opt, x, y, mask
            torch.cuda.empty_cache()
        except RuntimeError as e:
            good = False
            print(f"  {name} FAIL: {str(e)[:160]}")
            torch.cuda.empty_cache()

    # CHUNK_COMPILE on/off on Row 14 (the "optimization" delta)
    print("  -- CHUNK_COMPILE A/B on row14 --", flush=True)
    saved = B._CHUNK_COMPILE
    try:
        _, base, opt = build_models(32, 100000, 1024, 16, 1024, 2)
        del base
        x = make_input_fp16(32, 100000, 1024, seed=1234)
        mask = torch.ones(32, 100000, dtype=torch.bool, device=DEV)
        B._CHUNK_COMPILE = False
        med_off, _ = time_call(lambda: opt(x, mask), warmup=1, iters=3)
        B._CHUNK_COMPILE = True
        med_on, _ = time_call(lambda: opt(x, mask), warmup=2, iters=3)
        print(f"  row14  CHUNK_COMPILE=0 {med_off:8.1f} ms   "
              f"CHUNK_COMPILE=1 {med_on:8.1f} ms   "
              f"delta {100 * (med_off - med_on) / med_off:+.1f}%")
        del opt, x, mask
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  CHUNK_COMPILE A/B skipped: {str(e)[:160]}")
    finally:
        B._CHUNK_COMPILE = saved
        torch.cuda.empty_cache()
    return good


# --------------------------------------------------------------------------
# 5. row-14 accuracy: fp16-store vs fp32-store chunked
# --------------------------------------------------------------------------
@torch.no_grad()
def check_row14_accuracy():
    print("\n== 5. row14_accuracy ==", flush=True)
    for bs in (4, 2):
        try:
            _, base, opt = build_models(bs, 100000, 1024, 16, 1024, 2)
            del base
            x = make_input_fp16(bs, 100000, 1024, seed=1234)
            mask = torch.ones(bs, 100000, dtype=torch.bool, device=DEV)
            chk16 = opt._chunked_forward_causal(x.clone(), mask)
            ref32 = opt._chunked_forward_causal(x.clone().float(), mask,
                                                store=torch.float32)
            r = B.compare_outputs(ref32, chk16, RTOL, ATOL)
            p = r.failed_elements == 0
            print(f"  B={bs}: fp16-store vs fp32-store chunked  "
                  f"max_abs={r.max_abs_error:.3e} max_rel={r.max_relative_error:.3e} "
                  f"mean_abs={r.mean_abs_error:.3e} "
                  f"failed={r.failed_elements}/{r.total_elements}  "
                  f"{'PASS' if p else 'FAIL'}")
            del opt, x, mask, chk16, ref32
            torch.cuda.empty_cache()
            if not p:
                print("  CONTINGENCY: fp16 residual is over the 0.002 budget "
                      "at L=2. Do NOT ship this path -- needs fp32 residual + "
                      "hand-rolled block-flash (see RESUME.md).")
            return p
        except RuntimeError as e:
            print(f"  B={bs} OOM/err ({str(e)[:120]}) -- retrying smaller")
            torch.cuda.empty_cache()
    print("  FAIL: could not run row-14 accuracy at B=4 or B=2")
    return False


def main():
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free {gb(free):.2f} / {gb(total):.2f} GB  "
          f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")
    print(f"gate threshold CHUNK_ACT_ELEMS={B._CHUNK_ACT_ELEMS:,}  "
          f"CHUNK_MIN_SEQ={B._CHUNK_MIN_SEQ}  CHUNK_Q={B._CHUNK_Q}  "
          f"CHUNK_RESERVE_GB={B._CHUNK_RESERVE_GB}")

    check_sdpa_prefix_causal()                       # sys.exit(1) on failure
    results = {
        "gate": check_gate(),
        "equivalence": check_equivalence(),
        "oversize_capability": check_oversize_capability(),
        "row14_accuracy": check_row14_accuracy(),
    }
    print("\n== SUMMARY ==")
    for k, v in results.items():
        print(f"  {k:<22} {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
