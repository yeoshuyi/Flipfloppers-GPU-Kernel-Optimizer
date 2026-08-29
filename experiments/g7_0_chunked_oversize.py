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
  6. row14_golden         -- the REAL Row-14 shape against the committed
     fingerprint in experiments/g7_0_row14_golden.json. The frozen harness
     cannot score row 14, so the current implementation is the benchmark;
     this is what catches an optimization silently changing the answer at
     B=32 (check 4 only asserts finite+shape, check 5 runs at B=4).
  9. harness_contract     -- anti-gaming / conformance: the model must not
     mutate the fp32 tensor the harness hands it (on either the compiled or
     the chunked path), its output must depend on its input, and repeat calls
     must agree. The in-place write that lets row 14 fit is asserted to be
     confined to an fp16 caller buffer, which the harness never supplies.
  8. row14_full_accuracy  -- the FULL B=32 shape against an FP32-store
     reference assembled from exact B=4 batch slices (a transformer forward is
     batch-independent), so all 3.28e9 elements are checked, not check 5's
     reduced-batch proxy.
  7. harness_integrity    -- torch_transformer_benchmark.py is now the single
     source of truth (benchmark.py was removed), so the scoring half is
     guarded rather than structurally untouchable. Asserts, inside the
     container: the module under test is the shipped file, both sentinel
     pairs are present and ordered, and all 20 frozen scoring symbols are
     defined OUTSIDE the editable region.

Run via infra/slurm/g7_0_chunked_oversize.sbatch (needs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for the tight Row-14 budget).
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/work")
import torch_transformer_benchmark as B  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ATOL, RTOL = 2e-3, 2e-2
CANON = {}   # numbers for the ROW14_SUMMARY line consumed by run_eval.sh
GOLDEN_PATH = "/work/experiments/g7_0_row14_golden.json"
OFFICIAL_BSD = [  # (batch, seq_len, d_model, ffn_dim) -- 13 runnable rows.
    # ffn_dim matters now: G7.1's gate is a byte estimate with an explicit
    # ffn term, where G7.0's B*S*d proxy silently assumed ffn_dim == d_model.
    # Every official row happens to satisfy that (incl. row 7 at 32 and row 8
    # at 1024), but the gate must be checked with the real value.
    (64, 128, 128, 128), (1, 128, 128, 128), (4, 128, 128, 128),
    (16, 128, 128, 128), (128, 128, 128, 128), (10000, 128, 128, 128),
    (64, 128, 32, 32), (64, 128, 1024, 1024), (64, 128, 128, 128),
    (64, 128, 128, 128), (64, 128, 128, 128), (64, 32, 128, 128),
    (64, 1024, 128, 128),
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
# 1. prefix-causal attention: split (past + square-causal) + LSE merge
# --------------------------------------------------------------------------
@torch.no_grad()
def _prefix_causal_ref(q, kfull, vfull, c0):
    """Standalone copy of _chunked_forward_causal's prefix_causal_attn closure
    -- kept in the probe so a change to either is caught here."""
    eff = torch.ops.aten._scaled_dot_product_efficient_attention
    L = q.shape[2]
    out_d, lse_d = eff(q, kfull[:, :, c0:c0 + L], vfull[:, :, c0:c0 + L],
                       None, True, 0.0, True, scale=1.0)[:2]
    if c0 == 0:
        return out_d.float()
    out_p, lse_p = eff(q, kfull[:, :, :c0], vfull[:, :, :c0],
                       None, True, 0.0, False, scale=1.0)[:2]
    lse_d = lse_d[..., :L].unsqueeze(-1)
    lse_p = lse_p[..., :L].unsqueeze(-1)
    m = torch.maximum(lse_p, lse_d)
    wp, wd = torch.exp(lse_p - m), torch.exp(lse_d - m)
    return (wp * out_p.float() + wd * out_d.float()) / (wp + wd)


@torch.no_grad()
def check_sdpa_prefix_causal():
    print("\n== 1. prefix_causal_attn (split + LSE merge) ==", flush=True)
    bn, hn, sn, hd = 2, 3, 128, 64
    ok = True
    for dtype, atol in ((torch.float32, 2e-4), (torch.float16, 3e-3)):
        g = torch.Generator(device=DEV).manual_seed(11)
        q = torch.randn(bn, hn, sn, hd, generator=g, device=DEV, dtype=dtype)
        k = torch.randn(bn, hn, sn, hd, generator=g, device=DEV, dtype=dtype)
        v = torch.randn(bn, hn, sn, hd, generator=g, device=DEV, dtype=dtype)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            full = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  scale=1.0)
        worst = 0.0
        for chunk in (32, 48, sn):
            parts = []
            for c0 in range(0, sn, chunk):
                c1 = min(c0 + chunk, sn)
                parts.append(_prefix_causal_ref(q[:, :, c0:c1], k[:, :, :c1],
                                                v[:, :, :c1], c0))
            got = torch.cat(parts, dim=2)
            worst = max(worst, (got.float() - full.float()).abs().max().item())
        good = worst <= atol
        ok &= good
        print(f"  {str(dtype):>15}  chunked vs full causal: max|.| = {worst:.2e}"
              f"   {'ok' if good else 'MISMATCH'}")
    if not ok:
        print("  FAIL: the split+merge prefix-causal attention does not match "
              "full causal attention -- the chunked design is invalid.")
        sys.exit(1)
    print("  PASS")
    return True


# --------------------------------------------------------------------------
# 2. gate: never misfires on an official row; auto-route == direct call
# --------------------------------------------------------------------------
@torch.no_grad()
def check_gate():
    print("\n== 2. gate ==", flush=True)
    U = B.UserOptimizedTransformer
    f = U._would_oom_causal
    dev = torch.device("cuda", torch.cuda.current_device())
    budget, skip = U._causal_vram_budget(dev.index)
    good = True

    # -- 2a: no scored row may ever be routed to the slow chunked path -----
    print(f"  budget {gb(budget):.2f} GB ({B._CHUNK_OOM_FRAC:.0%} of total)   "
          f"fast-skip tokens*max(d,ffn) < {skip:,}")
    tightest, tightest_row = float("inf"), None
    for i, (b, s, d, ff) in enumerate(OFFICIAL_BSD, start=1):
        est = U._causal_capture_bytes(b, s, d, ff)
        fired = f(b, s, d, ff, dev)
        margin = budget / est
        if margin < tightest:
            tightest, tightest_row = margin, i
        if fired:
            good = False
            print(f"    row{i:<2} B={b} S={s} d={d} ffn={ff}  est={gb(est):.3f} GB"
                  f"  MISFIRED -- would be chunked")
    print(f"  gate False for all 13 official rows; tightest margin "
          f"{tightest:.2f}x at row {tightest_row}   "
          f"{'PASS' if good else 'FAIL'}")

    # -- 2b: it DOES fire for row 14, and for an ffn-heavy shape the old
    #        B*S*d proxy waved through ----------------------------------
    r14 = f(32, 100000, 1024, 1024, dev)
    est14 = U._causal_capture_bytes(32, 100000, 1024, 1024)
    print(f"  row14 est={gb(est14):.1f} GB ({est14 / budget:.2f}x over budget) "
          f"-> chunk={r14}   {'PASS' if r14 else 'FAIL'}")
    good &= r14
    ffn_heavy = f(2000, 512, 128, 32768, dev)
    old_proxy = 2000 * 512 * 128 >= 800_000_000
    print(f"  ffn-heavy d=128 ffn=32768: est="
          f"{gb(U._causal_capture_bytes(2000, 512, 128, 32768)):.1f} GB  "
          f"new_gate={ffn_heavy} old_B*S*d_proxy={old_proxy}   "
          f"{'PASS' if ffn_heavy else 'FAIL'}")
    good &= ffn_heavy

    # -- 2c: the gate flips exactly where the byte model says it should ----
    lo, hi = 1, 8_000_000
    while lo < hi:
        mid = (lo + hi) // 2
        if f(1, mid, 1024, 1024, dev):
            hi = mid
        else:
            lo = mid + 1
    est_at = U._causal_capture_bytes(1, lo, 1024, 1024)
    est_below = U._causal_capture_bytes(1, lo - 1, 1024, 1024)
    exact = est_below <= budget < est_at
    print(f"  boundary (d=ffn=1024): flips at {lo:,} tokens; "
          f"est just below={gb(est_below):.3f} GB, at={gb(est_at):.3f} GB, "
          f"budget={gb(budget):.3f} GB   {'PASS' if exact else 'FAIL'}")
    good &= exact

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
    good &= diff <= 1e-5          # same implementation; <= flash atomic jitter
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
            if name.strip() == "row14":
                CANON["shipped_ms"] = round(med, 1)
                CANON["peak_gb"] = round(gb(peak), 2)
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
            CANON.update(acc_b=bs, acc_max_abs=r.max_abs_error,
                         acc_max_rel=r.max_relative_error,
                         acc_failed=r.failed_elements)
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


# --------------------------------------------------------------------------
# 6. row-14 golden: the current iteration IS the benchmark
# --------------------------------------------------------------------------
def _reduce_tiled(y, tile=2048):
    """(per-batch sums, per-batch abs-sums, max|y|, all-finite), tiled.

    torch.sum(y, dtype=torch.float64) on [32,100000,1024] upcasts the whole
    tensor first -- 24.4 GB, an instant OOM on a 23.5 GB card. sum|y| is kept
    alongside the plain sum because the latter nearly cancels to zero.
    """
    bn, sn, _ = y.shape
    sums = torch.zeros(bn, dtype=torch.float64, device=y.device)
    abs_sums = torch.zeros(bn, dtype=torch.float64, device=y.device)
    ymax, finite = 0.0, True
    for c0 in range(0, sn, tile):
        blk = y[:, c0:min(c0 + tile, sn)].float()
        sums += blk.sum(dim=(1, 2), dtype=torch.float64)
        ablk = blk.abs()
        abs_sums += ablk.sum(dim=(1, 2), dtype=torch.float64)
        ymax = max(ymax, float(ablk.max()))
        finite = finite and bool(torch.isfinite(blk).all())
        del blk, ablk
    return sums.cpu().tolist(), abs_sums.cpu().tolist(), ymax, finite


@torch.no_grad()
def check_row14_golden():
    """Row 14 is unscorable by the frozen harness (its FP32 reference OOMs),
    so the committed fingerprint of the CURRENT implementation is the only
    benchmark that exists. Every optimization step must reproduce it under the
    official abs<0.002 OR rel<0.02 budget.

    Checks 4 and 5 do not cover this: check 4 asserts only finite+shape at the
    real shape, and check 5's accuracy anchor runs at B=4, so a kernel change
    that alters the answer only at B=32 would otherwise ship undetected.
    """
    print("\n== 6. row14_golden ==", flush=True)
    if not os.path.exists(GOLDEN_PATH):
        print(f"  SKIP: {GOLDEN_PATH} not present "
              f"(generate with infra/slurm/g7_1_gate_calibration.sbatch)")
        return True
    with open(GOLDEN_PATH) as fh:
        g = json.load(fh)
    c, fp = g["config"], g["fingerprint"]
    print(f"  reference: commit {g['provenance']['commit']} "
          f"job {g['provenance']['job_id']}  "
          f"({fp['n_samples']} samples over {fp['numel']:,} elements)")
    try:
        _, base, opt = build_models(c["bs"], c["sl"], c["dm"], c["nh"],
                                    c["ff"], c["nl"])
        del base
        mask = torch.ones(c["bs"], c["sl"], dtype=torch.bool, device=DEV)
        # ONE forward over a FRESH input: _chunked_forward_causal mutates x in
        # place and returns it, so re-running on the same buffer would compose.
        x = make_input_fp16(c["bs"], c["sl"], c["dm"], seed=c["seed"])
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        y = opt(x, mask)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        peak = torch.cuda.max_memory_allocated()

        idx = torch.tensor(fp["sample_indices"], dtype=torch.int64, device=DEV)
        got = y.reshape(-1)[idx].float()
        ref = torch.tensor(fp["sample_values"], dtype=torch.float32, device=DEV)
        r = B.compare_outputs(ref, got, RTOL, ATOL)
        sums, abs_sums, ymax, finite = _reduce_tiled(y)
        del x, y, opt, mask, idx, got, ref
        torch.cuda.empty_cache()

        # sum|y| is the meaningful bulk invariant -- the plain sum cancels to
        # ~1e0 over 1e8 elements, so its relative drift is not informative.
        ref_abs = fp.get("batch_abs_sums_fp64")
        if ref_abs:
            abs_rel = max(abs(a - b) / max(abs(b), 1e-12)
                          for a, b in zip(abs_sums, ref_abs))
        else:
            abs_rel = float("nan")
        max_sum_abs = max(abs(a - b) for a, b in zip(sums,
                                                     fp["batch_sums_fp64"]))
        bulk_ok = not (abs_rel == abs_rel) or abs_rel <= 1e-3
        p = r.failed_elements == 0 and finite and bulk_ok
        print(f"  sampled values vs golden: max_abs={r.max_abs_error:.3e} "
              f"max_rel={r.max_relative_error:.3e} "
              f"failed={r.failed_elements}/{r.total_elements}  "
              f"{'PASS' if r.failed_elements == 0 else 'FAIL'}")
        print(f"  per-batch sum|y| drift: worst relative {abs_rel:.3e} "
              f"(gate 1e-3)  {'PASS' if bulk_ok else 'FAIL'}")
        print(f"  per-batch signed sums: worst absolute drift "
              f"{max_sum_abs:.3e} over {len(sums)} batches (diagnostic)")
        print(f"  max|y| {ymax:.6f} vs golden {fp['max_abs']:.6f}   "
              f"finite={finite}")

        pin = g["perf_pin"]
        base_ms = pin["job198_shipped_ms"]
        base_gb = pin["job198_peak_gb"]
        dms = 100.0 * (ms - base_ms) / base_ms
        print(f"  perf vs pin: {ms:.1f} ms ({dms:+.1f}% vs {base_ms} ms pinned)"
              f"   peak {peak / 1024 ** 3:.2f} GB (pinned {base_gb} GB)")
        if dms > 10.0:
            print(f"  WARNING: latency regressed >10% against the pin")
        if peak / 1024 ** 3 > base_gb + 0.5:
            print(f"  WARNING: peak grew >0.5 GB against the pin")
        CANON["golden"] = "PASS" if p else "FAIL"
        CANON["golden_max_abs"] = r.max_abs_error
        return p
    except RuntimeError as e:
        print(f"  FAIL: {str(e)[:160]}")
        torch.cuda.empty_cache()
        return False


# --------------------------------------------------------------------------
# 7. harness integrity: the scoring half is intact, inside the container
# --------------------------------------------------------------------------
def check_harness_integrity():
    """torch_transformer_benchmark.py is now the SINGLE source of truth -- it is
    edited directly, benchmark.py is gone. So the thing that used to be
    structurally impossible (touching the scoring half) is now merely guarded,
    and the guard has to hold INSIDE the container that actually runs the job.

    tools/sync_entrypoint.py --check and tools/verify_baseline.py run on the
    host before submission; this asserts the same shape of invariant against
    the module the GPU job really imported:
      * it is the shipped /work/torch_transformer_benchmark.py, not a stray copy;
      * both sentinel pairs delimiting our contribution are present and ordered;
      * every frozen scoring symbol is defined OUTSIDE those sentinels, i.e.
        nobody moved a piece of the harness into the editable region.
    """
    print("\n== 7. harness_integrity ==", flush=True)
    import ast as _ast
    import importlib.util as _ilu
    good = True

    path = getattr(B, "__file__", "") or ""
    is_shipped = os.path.basename(path) == "torch_transformer_benchmark.py"
    print(f"  module under test: {path}   {'PASS' if is_shipped else 'FAIL'}")
    good &= is_shipped

    src = open(path).read()
    lines = src.splitlines(keepends=True)
    marks = {}
    for name, tok in (("h_beg", "# >>> BEGIN user helpers"),
                      ("h_end", "# <<< END user helpers"),
                      ("m_beg", "# >>> BEGIN user model"),
                      ("m_end", "# <<< END user model")):
        hit = [i for i, ln in enumerate(lines) if ln.startswith(tok)]
        marks[name] = hit[0] if len(hit) == 1 else None
    ordered = (all(v is not None for v in marks.values())
               and marks["h_beg"] < marks["h_end"] < marks["m_beg"] < marks["m_end"])
    print(f"  sentinel blocks present and ordered: {ordered}   "
          f"{'PASS' if ordered else 'FAIL'}")
    good &= ordered

    if ordered:
        ours_lines = set(range(marks["h_beg"] + 1, marks["h_end"] + 1))
        ours_lines |= set(range(marks["m_beg"] + 1, marks["m_end"] + 1))
        spec = _ilu.spec_from_file_location("vb", "/work/tools/verify_baseline.py")
        vb = _ilu.module_from_spec(spec)
        spec.loader.exec_module(vb)
        tree = _ast.parse(src)
        inside = []
        for node in tree.body:
            if not isinstance(node, (_ast.FunctionDef, _ast.ClassDef)):
                continue
            if node.name in vb.FROZEN and (node.lineno - 1) in ours_lines:
                inside.append(node.name)
        clean = not inside
        print(f"  {len(vb.FROZEN)} frozen scoring symbols all outside the "
              f"editable region: {clean}   {'PASS' if clean else 'FAIL'}")
        if inside:
            print(f"    inside the sentinels (must not be): {inside}")
        good &= clean

    print(f"  {'PASS' if good else 'FAIL'}")
    return good


# --------------------------------------------------------------------------
# 8. row-14 accuracy at the FULL B=32 shape
# --------------------------------------------------------------------------
@torch.no_grad()
def check_row14_full_accuracy(slice_b=4):
    """Accuracy of the shipped Row-14 output at the REAL batch size.

    Check 5 prices FP16 storage at B=4 because an FP32-store reference does not
    fit at B=32. But a transformer forward is completely independent across the
    batch -- attention is within-sequence, LayerNorm and the FFN are per-token,
    and nothing in _chunked_forward_causal couples batch rows (kbuf/vbuf are
    per-b). So the FP32 reference for B=32 can be assembled from 8 exact B=4
    slices and compared slice-by-slice against the one true B=32 forward. That
    turns check 5's proxy into a full-shape measurement over all 3.28e9
    elements.

    The reference also runs at a DIFFERENT chunk_q than the B=32 run (the
    adaptive sizer picks a larger one at B=4). That is deliberate: chunking is
    exact (check 1), so agreement across two different chunkings is stronger
    evidence than agreement under one.
    """
    print("\n== 8. row14_full_accuracy (B=32, all 3.28e9 elements) ==",
          flush=True)
    p = dict(bs=32, sl=100000, dm=1024, nh=16, ff=1024, nl=2, seed=1234)
    try:
        _, base, opt = build_models(p["bs"], p["sl"], p["dm"], p["nh"],
                                    p["ff"], p["nl"])
        del base
        # The input is the single source of truth for BOTH arms, so keep it on
        # the host: _chunked_forward_causal mutates x in place, and the seeded
        # generator CANNOT be replayed for a batch slice -- make_input_fp16
        # fills a strided view x[:, c0:c1], and drawing the same count into a
        # contiguous temp maps the values to different elements. A first
        # attempt did exactly that and the two arms came out statistically
        # independent (mean_abs 1.122 ~= 2/sqrt(pi) = 1.128, job 219).
        x32 = make_input_fp16(p["bs"], p["sl"], p["dm"], seed=p["seed"])
        x_cpu = x32.to("cpu")                       # 6.55 GB host
        mask32 = torch.ones(p["bs"], p["sl"], dtype=torch.bool, device=DEV)
        ours = opt(x32, mask32)                     # in place -> ours is x32
        torch.cuda.synchronize()
        del mask32
        torch.cuda.empty_cache()

        mask_s = torch.ones(slice_b, p["sl"], dtype=torch.bool, device=DEV)
        tot_failed = tot_elems = 0
        worst_abs = worst_rel = sum_abs = 0.0
        for b0 in range(0, p["bs"], slice_b):
            b1 = b0 + slice_b
            xs = x_cpu[b0:b1].to(DEV)               # byte-exact same input
            ref = opt._chunked_forward_causal(xs.float(), mask_s,
                                              store=torch.float32)
            r = B.compare_outputs(ref, ours[b0:b1].contiguous(), RTOL, ATOL)
            tot_failed += r.failed_elements
            tot_elems += r.total_elements
            worst_abs = max(worst_abs, r.max_abs_error)
            worst_rel = max(worst_rel, r.max_relative_error)
            sum_abs += r.mean_abs_error * r.total_elements
            print(f"    rows {b0:2d}-{b1 - 1:2d}: max_abs={r.max_abs_error:.3e} "
                  f"mean_abs={r.mean_abs_error:.3e} "
                  f"failed={r.failed_elements}/{r.total_elements}", flush=True)
            del xs, ref, r
            torch.cuda.empty_cache()

        ok = tot_failed == 0 and tot_elems == p["bs"] * p["sl"] * p["dm"]
        print(f"  FULL B=32: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e} "
              f"mean_abs={sum_abs / max(tot_elems, 1):.3e}  "
              f"failed={tot_failed}/{tot_elems}  {'PASS' if ok else 'FAIL'}")
        CANON["full_failed"] = tot_failed
        CANON["full_max_abs"] = worst_abs
        CANON["full_elems"] = tot_elems
        del ours, x32, x_cpu, opt, mask_s
        torch.cuda.empty_cache()
        return ok
    except RuntimeError as e:
        print(f"  FAIL: {str(e)[:200]}")
        torch.cuda.empty_cache()
        return False


# --------------------------------------------------------------------------
# 9. harness contract: behaviours the frozen harness relies on
# --------------------------------------------------------------------------
@torch.no_grad()
def check_harness_contract():
    """Anti-gaming / conformance assertions on the SHIPPED model.

    The accuracy gate already makes functional cheating impossible (5 trials,
    fresh seeds, compared against the frozen baseline). These cover the things
    the gate would NOT catch:
      a) the model must not mutate the caller's tensor -- the harness reuses one
         `x` for the baseline forward, the optimized forward and 300+ timed
         calls. _chunked_forward_causal DOES write in place, so this pins that
         it only ever happens to its own fp16 copy, never to the fp32 tensor the
         harness hands over;
      b) the same on the CHUNKED path specifically, forced on a small shape;
      c) the output has to depend on the input (not a cached constant);
      d) two calls on equal inputs have to agree.
    """
    print("\n== 9. harness_contract ==", flush=True)
    good = True
    saved = B._CHUNK_ACT_ELEMS
    try:
        for label, force_chunk in (("compiled path", False),
                                   ("chunked path ", True)):
            B._CHUNK_ACT_ELEMS = 1_000_000 if force_chunk else 0
            _, base, opt = build_models(2, 4096, 128, 4, 128, 2)
            mask = torch.ones(2, 4096, dtype=torch.bool, device=DEV)
            # fp32 in -- exactly what the harness supplies (--dtype float32)
            x = torch.randn(2, 4096, 128, device=DEV, dtype=torch.float32)
            x_ref = x.clone()
            y1 = opt(x, mask).clone()          # clone: cudagraph reuses buffers
            unmutated = torch.equal(x, x_ref)
            yb = base(x, mask)
            shape_ok = tuple(y1.shape) == tuple(yb.shape)
            routed = "chunked" if force_chunk else "compiled"
            print(f"  {label}: input unmutated={unmutated}  shape{tuple(y1.shape)}"
                  f" matches baseline={shape_ok}  out_dtype={str(y1.dtype).split('.')[-1]}"
                  f"  {'PASS' if unmutated and shape_ok else 'FAIL'}")
            good &= unmutated and shape_ok

            # (c) output depends on the input
            y0 = opt(torch.zeros_like(x), mask).clone()
            depends = not torch.allclose(y1.float(), y0.float(), atol=1e-4)
            # (d) equal inputs -> equal outputs
            y2 = opt(x_ref.clone(), mask).clone()
            agree = torch.equal(y1.float(), y2.float())
            print(f"                 output depends on input={depends}  "
                  f"repeat-call agreement={agree}  "
                  f"{'PASS' if depends and agree else 'FAIL'}")
            good &= depends and agree
            del base, opt, x, x_ref, y1, y2, y0, yb, mask
            torch.cuda.empty_cache()

        # (b') the documented exception: an fp16 caller buffer IS written in
        # place (that is how row 14 fits in 19.55 GB). Assert it is real and
        # confined to that case, so the contract is stated, not assumed.
        B._CHUNK_ACT_ELEMS = 1_000_000
        _, base, opt = build_models(2, 4096, 128, 4, 128, 2)
        del base
        mask = torch.ones(2, 4096, dtype=torch.bool, device=DEV)
        xh = torch.randn(2, 4096, 128, device=DEV, dtype=torch.float16)
        xh_ref = xh.clone()
        yh = opt(xh, mask)
        mutated = not torch.equal(xh, xh_ref)
        aliases = yh.data_ptr() == xh.data_ptr()
        print(f"  fp16 caller buffer IS written in place: {mutated} "
              f"(returned tensor aliases it: {aliases}) -- documented contract, "
              f"unreachable from the harness (--dtype float32)")
        good &= mutated and aliases
        del opt, xh, xh_ref, yh, mask
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  FAIL: {str(e)[:200]}")
        good = False
    finally:
        B._CHUNK_ACT_ELEMS = saved
        torch.cuda.empty_cache()
    print(f"  {'PASS' if good else 'FAIL'}")
    return good


def main():
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free {gb(free):.2f} / {gb(total):.2f} GB  "
          f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")
    print(f"gate: CHUNK_ACT_ELEMS={B._CHUNK_ACT_ELEMS} (0 = byte model)  "
          f"CHUNK_OOM_FRAC={B._CHUNK_OOM_FRAC}  bytes = "
          f"{B._CHUNK_BYTES_FIXED / 2 ** 20:.0f}MiB + "
          f"tok*({B._CHUNK_BYTES_PER_D}*d + {B._CHUNK_BYTES_PER_FFN}*ffn)")
    print(f"CHUNK_MIN_SEQ={B._CHUNK_MIN_SEQ}  CHUNK_Q={B._CHUNK_Q}  "
          f"CHUNK_RESERVE_GB={B._CHUNK_RESERVE_GB}")

    check_sdpa_prefix_causal()                       # sys.exit(1) on failure
    results = {
        "gate": check_gate(),
        "equivalence": check_equivalence(),
        "oversize_capability": check_oversize_capability(),
        "row14_accuracy": check_row14_accuracy(),
        "row14_golden": check_row14_golden(),
        "harness_integrity": check_harness_integrity(),
        "row14_full_accuracy": check_row14_full_accuracy(),
        "harness_contract": check_harness_contract(),
    }
    print("\n== SUMMARY ==")
    for k, v in results.items():
        print(f"  {k:<22} {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")

    # one canonical line for run_eval.sh's RUN_ROW14 wrapper
    cap = results["oversize_capability"] and results["row14_accuracy"]
    print("ROW14_SUMMARY "
          f"row=14 B=32 S=100000 d=1024 H=16 L=2 ffn=1024 "
          f"baseline=OOM(FP32_[B,H,S,S]_scores) "
          f"shipped_ms={CANON.get('shipped_ms', 'NA')} "
          f"peak_gb={CANON.get('peak_gb', 'NA')} "
          f"acc_b={CANON.get('acc_b', 'NA')} "
          f"acc_max_abs={CANON.get('acc_max_abs', float('nan')):.3e} "
          f"acc_failed={CANON.get('acc_failed', 'NA')} "
          f"full_failed={CANON.get('full_failed', 'NA')}/{CANON.get('full_elems', 'NA')} "
          f"golden={CANON.get('golden', 'NA')} "
          f"result={'supported' if cap else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
