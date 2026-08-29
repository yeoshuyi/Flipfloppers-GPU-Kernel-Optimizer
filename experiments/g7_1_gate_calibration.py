"""
G7.1 -- (a) calibrate the compiled causal path's activation footprint so
_would_oom_causal can be restated in BYTES against real device capacity, and
(b) capture the Row-14 golden fingerprint that pins "our current iteration" as
the benchmark for the optimization arc.

Why (a): the shipped gate is `B*S*d >= 8e8` -- an element count encoding an
undocumented "~25 bytes/elem" model, blind to ffn_dim and to the actual card.
The replacement gate has NO OOM fallback, so its byte model must never
UNDER-predict. This measures predicted-vs-actual peak across shapes that vary
tokens, d_model, ffn_dim and num_layers independently, and fits

    activation_bytes ~= tokens * (C_d * d_model + C_ffn * ffn_dim)

Why (b): the frozen harness cannot score Row 14 (its FP32 reference OOMs), and
probe check 4 asserts only `finite` + `shape_ok` at the real shape. Nothing
pins the ANSWER. This writes experiments/g7_0_row14_golden.json -- per-batch
fp64 sums plus values at a fixed seeded index set -- so every later step (A1
flash swap, A2 relayout, D/B/C) must reproduce it under the official
abs<0.002 OR rel<0.02 budget.

NOTE _chunked_forward_causal mutates x IN PLACE and returns that same tensor
(benchmark.py:1434). The golden is therefore ONE application to a FRESH input,
not the 5-applications-to-one-buffer that check 4's timing loop happens to do.

Run via infra/slurm/g7_1_gate_calibration.sbatch. sbatch only, never direct
GPU python.
"""
import gc
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, "/work")
import benchmark as B  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

GOLDEN_PATH = "/work/experiments/g7_0_row14_golden.json"
N_SAMPLES = 8192
SAMPLE_SEED = 20260829
ROW14 = dict(bs=32, sl=100000, dm=1024, nh=16, ff=1024, nl=2, seed=1234)


def gb(nbytes):
    return nbytes / 1024 ** 3


def build_opt(bs, sl, dm, nh, ff, nl):
    """Optimized model only, weights copied from a fresh baseline (seeded)."""
    cfg = B.TransformerConfig(batch_size=bs, seq_len=sl, d_model=dm,
                              num_heads=nh, ffn_dim=ff, num_layers=nl,
                              causal=True)
    cfg.validate()
    torch.manual_seed(0)
    base = B.BaselineTransformer(cfg).to(DEV, torch.float32).eval()
    opt = B.UserOptimizedTransformer(cfg).to(DEV, torch.float32).eval()
    B.copy_model_weights(base, opt, strict=True)
    del base
    gc.collect()
    return cfg, opt


def make_input_fp16(bn, sn, dn, seed, tile=8192):
    """Identical to the g7_0 probe's helper -- the golden depends on it."""
    x = torch.empty(bn, sn, dn, dtype=torch.float16, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for c0 in range(0, sn, tile):
        x[:, c0:min(c0 + tile, sn)].normal_(0.0, 1.0, generator=g)
    return x


def hard_reset():
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------
# (a) calibration of the COMPILED causal path
# --------------------------------------------------------------------------
# (label, batch, seq_len, d_model, heads, ffn_dim, layers)
OFFICIAL = [
    ("row01", 64, 128, 128, 4, 128, 4),
    ("row02", 1, 128, 128, 4, 128, 4),
    ("row03", 4, 128, 128, 4, 128, 4),
    ("row04", 16, 128, 128, 4, 128, 4),
    ("row05", 128, 128, 128, 4, 128, 4),
    ("row06", 10000, 128, 128, 4, 128, 4),
    ("row07", 64, 128, 32, 4, 32, 4),
    ("row08", 64, 128, 1024, 4, 1024, 4),
    ("row09", 64, 128, 128, 1, 128, 4),
    ("row10", 64, 128, 128, 2, 128, 4),
    ("row11", 64, 128, 128, 16, 128, 4),
    ("row12", 64, 32, 128, 4, 128, 4),
    ("row13", 64, 1024, 128, 4, 128, 4),
]

# d != ffn separates C_d from C_ffn (every official row has d == ffn);
# the b-ladder confirms linearity in tokens; the L-ladder confirms the peak is
# ~flat in num_layers (allocator reuse across the unrolled loop).
SYNTHETIC = [
    ("dneq_d128_f1024", 400, 512, 128, 4, 1024, 4),
    ("dneq_d128_f4096", 400, 512, 128, 4, 4096, 4),
    ("dneq_d256_f2048", 400, 512, 256, 4, 2048, 4),
    ("dneq_d1024_f128", 400, 512, 1024, 16, 128, 4),
    ("dneq_d1024_f256", 200, 512, 1024, 16, 256, 4),
    ("tok_b1000", 1000, 128, 128, 4, 128, 4),
    ("tok_b2000", 2000, 128, 128, 4, 128, 4),
    ("tok_b4000", 4000, 128, 128, 4, 128, 4),
    ("tok_b8000", 8000, 128, 128, 4, 128, 4),
    ("lay_L2", 2000, 128, 128, 4, 128, 2),
    ("lay_L8", 2000, 128, 128, 4, 128, 8),
]


@torch.no_grad()
def measure_peak(label, bs, sl, dm, nh, ff, nl):
    """Activation bytes of the compiled causal path, above the weights+input
    floor. Returns (capture_delta, steady_delta) or None if the shape errored.

    capture = includes the torch.compile / cudagraph capture pass (this is what
    the gate actually has to survive); steady = warmed-up replay.
    """
    hard_reset()
    try:
        cfg, opt = build_opt(bs, sl, dm, nh, ff, nl)
        x = torch.randn(bs, sl, dm, device=DEV, dtype=torch.float32)
        mask = torch.ones(bs, sl, dtype=torch.bool, device=DEV)

        # the gate must NOT fire here -- we are calibrating the compiled path
        assert not opt._would_oom_causal(bs, sl, dm), \
            f"{label}: gate fired on a calibration shape"

        floor = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        y = opt(x, mask)                       # compile + capture
        torch.cuda.synchronize()
        capture_peak = torch.cuda.max_memory_allocated()
        finite = bool(torch.isfinite(y).all())
        del y

        for _ in range(2):                     # settle into replay
            y = opt(x, mask)
            del y
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            y = opt(x, mask)
            del y
        torch.cuda.synchronize()
        steady_peak = torch.cuda.max_memory_allocated()

        tokens = bs * sl
        cap_d = capture_peak - floor
        std_d = steady_peak - floor
        print(f"  {label:<16} tok={tokens:>9,} d={dm:<5} ffn={ff:<5} L={nl}  "
              f"floor={gb(floor):6.3f}  capture={gb(cap_d):7.3f} GB  "
              f"steady={gb(std_d):7.3f} GB  "
              f"B/tok: cap={cap_d / tokens:8.1f} std={std_d / tokens:8.1f}  "
              f"finite={finite}", flush=True)
        del opt, x, mask, cfg
        hard_reset()
        return dict(label=label, tokens=tokens, d=dm, ffn=ff, layers=nl,
                    capture=cap_d, steady=std_d, floor=floor)
    except RuntimeError as e:
        print(f"  {label:<16} ERROR: {str(e)[:150]}", flush=True)
        hard_reset()
        return None


def fit_coeffs(rows, key):
    """Least-squares fit of  bytes/token = C_d*d + C_ffn*ffn  (no intercept)."""
    import numpy as np
    A = np.array([[r["d"], r["ffn"]] for r in rows], dtype=np.float64)
    y = np.array([r[key] / r["tokens"] for r in rows], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def report_fit(rows, key, c_d, c_ffn):
    print(f"\n  -- fit on '{key}':  bytes/token = {c_d:.2f}*d + {c_ffn:.2f}*ffn")
    worst = 0.0
    print(f"     {'label':<16} {'measured':>10} {'predicted':>10} {'pred/meas':>10}")
    for r in rows:
        meas = r[key] / r["tokens"]
        pred = c_d * r["d"] + c_ffn * r["ffn"]
        ratio = pred / meas if meas > 0 else float("inf")
        worst = max(worst, meas / pred) if pred > 0 else float("inf")
        flag = "  <-- UNDER-PREDICTS" if ratio < 1.0 else ""
        print(f"     {r['label']:<16} {meas:10.1f} {pred:10.1f} {ratio:10.3f}{flag}")
    print(f"     worst measured/predicted = {worst:.3f} "
          f"(>1.0 means the fit under-predicts somewhere)")
    return worst


def calibrate():
    print("\n" + "=" * 74)
    print("(a) CALIBRATION -- compiled causal path activation footprint")
    print("=" * 74, flush=True)
    rows = []
    print("\n-- official rows 1-13 (all have ffn_dim == d_model) --", flush=True)
    for spec in OFFICIAL:
        r = measure_peak(*spec)
        if r:
            rows.append(r)
    print("\n-- synthetic: d != ffn, token ladder, layer ladder --", flush=True)
    for spec in SYNTHETIC:
        r = measure_peak(*spec)
        if r:
            rows.append(r)

    if len(rows) < 4:
        print("\n  too few successful shapes to fit")
        return rows, None

    print("\n" + "-" * 74)
    fits = {}
    for key in ("capture", "steady"):
        c_d, c_ffn = fit_coeffs(rows, key)
        worst = report_fit(rows, key, c_d, c_ffn)
        fits[key] = (c_d, c_ffn, worst)

    # The shipped coefficients come from the CAPTURE fit (the pass the gate
    # must survive), scaled so predicted >= measured everywhere with >=1.25x
    # headroom -- the no-OOM-fallback decision requires never under-predicting.
    c_d, c_ffn, worst = fits["capture"]
    scale = max(1.0, worst) * 1.25
    print(f"\n  SHIPPED (capture fit x {scale:.3f} for >=1.25x headroom):")
    print(f"    C_d   = {c_d * scale:.1f}  -> round up to {int(c_d * scale) + 1}")
    print(f"    C_ffn = {c_ffn * scale:.1f}  -> round up to {int(c_ffn * scale) + 1}")
    print(f"  naive live-set upper bound from the source is 36*d + 10*ffn")
    return rows, (c_d * scale, c_ffn * scale)


# --------------------------------------------------------------------------
# (b) Row-14 golden fingerprint
# --------------------------------------------------------------------------
def sample_indices(numel, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, numel, (n,), generator=g, dtype=torch.int64)


@torch.no_grad()
def capture_golden():
    print("\n" + "=" * 74)
    print("(b) ROW-14 GOLDEN FINGERPRINT")
    print("=" * 74, flush=True)
    hard_reset()
    p = ROW14
    cfg, opt = build_opt(p["bs"], p["sl"], p["dm"], p["nh"], p["ff"], p["nl"])
    mask = torch.ones(p["bs"], p["sl"], dtype=torch.bool, device=DEV)

    # --- the reference application: ONE forward over a FRESH input ---------
    x = make_input_fp16(p["bs"], p["sl"], p["dm"], seed=p["seed"])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    y = opt(x, mask)
    torch.cuda.synchronize()
    single_ms = (time.perf_counter() - t0) * 1e3
    peak_single = torch.cuda.max_memory_allocated()

    numel = y.numel()
    idx = sample_indices(numel, N_SAMPLES, SAMPLE_SEED)
    samples = y.reshape(-1)[idx.to(DEV)].float().cpu().tolist()
    batch_sums = torch.sum(y, dim=(1, 2), dtype=torch.float64).cpu().tolist()
    finite = bool(torch.isfinite(y).all())
    ymax = float(y.abs().max().float())
    print(f"  one forward on fresh input: {single_ms:.1f} ms  "
          f"peak={gb(peak_single):.2f} GB  finite={finite}  max|y|={ymax:.4f}")
    print(f"  fingerprint: {len(batch_sums)} per-batch fp64 sums + "
          f"{len(samples)} sampled values (seed {SAMPLE_SEED}) over "
          f"{numel:,} elements")
    del x, y
    hard_reset()

    # --- perf pin, matching check 4's methodology exactly ------------------
    # (check 4 reuses ONE buffer across warmup+iters, feeding each output back
    # in as the next input; replicated here so the number is comparable to the
    # 13018.2 ms / 20.80 GB recorded in job 198.)
    x = make_input_fp16(p["bs"], p["sl"], p["dm"], seed=p["seed"])
    torch.cuda.reset_peak_memory_stats()
    opt(x, mask)
    torch.cuda.synchronize()
    ts = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt(x, mask)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    med_ms = ts[len(ts) // 2]
    peak_loop = torch.cuda.max_memory_allocated()
    print(f"  check-4-style timing: median={med_ms:.1f} ms "
          f"(all {['%.0f' % t for t in ts]})  peak={gb(peak_loop):.2f} GB")
    del x, mask, opt, cfg
    hard_reset()

    try:
        commit = subprocess.check_output(
            ["git", "-C", "/work", "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        commit = "unknown"

    golden = {
        "_comment": "Row-14 golden reference. The frozen harness cannot score "
                    "row 14 (its FP32 path OOMs), so THIS is the benchmark: "
                    "every later optimization must reproduce these values "
                    "under abs<0.002 OR rel<0.02. Regenerate ONLY when "
                    "deliberately rebaselining.",
        "config": dict(p),
        "reference": {
            "applications": 1,
            "note": "_chunked_forward_causal mutates x in place and returns "
                    "it; this is ONE forward over a FRESH seeded input.",
        },
        "provenance": {
            "commit": commit,
            "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"),
        },
        "perf_pin": {
            "job198_shipped_ms": 13018.2,
            "job198_peak_gb": 20.80,
            "this_job_single_ms": round(single_ms, 1),
            "this_job_single_peak_gb": round(gb(peak_single), 2),
            "this_job_check4_median_ms": round(med_ms, 1),
            "this_job_check4_peak_gb": round(gb(peak_loop), 2),
        },
        "fingerprint": {
            "numel": numel,
            "finite": finite,
            "max_abs": ymax,
            "sample_seed": SAMPLE_SEED,
            "n_samples": N_SAMPLES,
            "sample_indices": idx.tolist(),
            "sample_values": samples,
            "batch_sums_fp64": batch_sums,
        },
    }
    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=1)
    print(f"  wrote {GOLDEN_PATH} "
          f"({os.path.getsize(GOLDEN_PATH) / 1024:.0f} KB)")
    return golden


def main():
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM free {gb(free):.2f} / {gb(total):.2f} GB  "
          f"total_memory={gb(props.total_memory):.3f} GB  "
          f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")
    print(f"current gate: CHUNK_ACT_ELEMS={B._CHUNK_ACT_ELEMS:,}")

    rows, shipped = calibrate()
    golden = capture_golden()

    print("\n== SUMMARY ==")
    if shipped:
        print(f"  proposed byte model: {shipped[0]:.1f}*d + {shipped[1]:.1f}*ffn "
              f"bytes per token")
    print(f"  golden written for commit "
          f"{golden['provenance']['commit']} job "
          f"{golden['provenance']['job_id']}")
    print("\nG7_1_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
