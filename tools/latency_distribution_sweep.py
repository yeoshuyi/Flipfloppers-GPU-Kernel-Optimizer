#!/usr/bin/env python3
"""Sweep the official 14-row causal shape matrix (CLAUDE.md -> "OFFICIAL CAUSAL
EVALUATION MATRIX") collecting raw per-iteration latency samples per
(shape, variant), for studying how latency *varies* at each shape -- not for
scoring speedup. That's what torch_transformer_benchmark.py's own
benchmark_models() does; this script instead dumps every timed sample.

Reuses torch_transformer_benchmark.py's own timing primitives
(generate_random_case, warmup_model, benchmark_once, resolve_device/dtype) by
importing it as a module. Never edits or re-splices that file -- it's the
frozen scoring harness (see tools/sync_entrypoint.py / tools/verify_baseline.py).

Row 14 (B=32, d_model=1024, heads=16, seq_len=100000, layers=2, ffn_dim=1024)
is swept optimized-only: the frozen BaselineTransformer OOMs on this shape
before any math runs (see infra/slurm/official_causal_sweep.sbatch), so it is
never attempted here either. It also needs a process of its own: a default
run (below) skips it and prints a reminder to run it as a SEPARATE, later
invocation with --only-row 14. This isn't optional plumbing -- a still-running
sibling process's CUDA context (even after empty_cache(), a few GiB stays
resident for the process's lifetime: cuBLAS/cuDNN handles, torch.compile's
caches) is enough to OOM row 14's ~12 GiB kv-cache allocation on this 24 GiB
card (confirmed empirically); only an actually-exited prior process fully
frees its share back to the device.

    python3 tools/latency_distribution_sweep.py --out results/artifacts/x_rows1-13.json
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        python3 tools/latency_distribution_sweep.py --only-row 14 --out results/artifacts/x_row14.json

tools/plot_latency_distribution.py accepts multiple JSON files and merges
their series, so rows 1-13 and row 14 can be plotted from separate files.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import torch_transformer_benchmark as tb  # noqa: E402

# Transcribed verbatim from CLAUDE.md "OFFICIAL CAUSAL EVALUATION MATRIX"
# (lines 206-222) / run_eval.sh ROWS (lines 44-58). All rows are causal=True.
OFFICIAL_ROWS = [
    dict(row=1, batch_size=64, d_model=128, heads=4, seq_len=128, layers=4, ffn_dim=128),
    dict(row=2, batch_size=1, d_model=128, heads=4, seq_len=128, layers=4, ffn_dim=128),
    dict(row=3, batch_size=4, d_model=128, heads=4, seq_len=128, layers=4, ffn_dim=128),
    dict(row=4, batch_size=16, d_model=128, heads=4, seq_len=128, layers=4, ffn_dim=128),
    dict(row=5, batch_size=128, d_model=128, heads=4, seq_len=128, layers=4, ffn_dim=128),
    dict(row=6, batch_size=10000, d_model=128, heads=4, seq_len=128, layers=4, ffn_dim=128),
    dict(row=7, batch_size=64, d_model=32, heads=4, seq_len=128, layers=4, ffn_dim=32),
    dict(row=8, batch_size=64, d_model=1024, heads=4, seq_len=128, layers=4, ffn_dim=1024),
    dict(row=9, batch_size=64, d_model=128, heads=1, seq_len=128, layers=4, ffn_dim=128),
    dict(row=10, batch_size=64, d_model=128, heads=2, seq_len=128, layers=4, ffn_dim=128),
    dict(row=11, batch_size=64, d_model=128, heads=16, seq_len=128, layers=4, ffn_dim=128),
    dict(row=12, batch_size=64, d_model=128, heads=4, seq_len=32, layers=4, ffn_dim=128),
    dict(row=13, batch_size=64, d_model=128, heads=4, seq_len=1024, layers=4, ffn_dim=128),
    dict(row=14, batch_size=32, d_model=1024, heads=16, seq_len=100000, layers=2, ffn_dim=1024),
]
ROW14 = 14


def gpu_probe() -> Optional[dict]:
    """Best-effort clocks/temp snapshot, per the anti-noise protocol in docs/SETUP.md."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        sm_clock, temp = (p.strip() for p in out.split(","))
        return {"sm_clock_mhz": float(sm_clock), "temperature_c": float(temp)}
    except Exception:
        return None


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def make_row14_input(row: dict, device: torch.device, seed: int, tile: int = 8192):
    """Row 14's [32,100000,1024] input, built directly in fp16 and in sequence
    tiles so there is never an fp32 -- or even a 2x fp16 -- transient. Mirrors
    experiments/g7_0_chunked_oversize.py's make_input_fp16() exactly, which is
    the proven way this project exercises row 14's latency at all: calling
    tb.generate_random_case() here (dtype=float32, or its `x = x * input_scale`
    doubling even at fp16) OOMs a 23.52 GiB (usable) RTX 4090 before the model
    ever runs -- confirmed empirically, and consistent with why the frozen
    harness can't score row 14 (infra/slurm/official_causal_sweep.sbatch lines
    96-114). The model itself stays fp32 like every other row -- only the
    input is fp16, which is what lets it fit; the chunked forward internally
    upcasts per chunk as it goes."""
    x = torch.empty(row["batch_size"], row["seq_len"], row["d_model"], dtype=torch.float16, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    for c0 in range(0, row["seq_len"], tile):
        x[:, c0 : min(c0 + tile, row["seq_len"])].normal_(0.0, 1.0, generator=g)
    valid_mask = torch.ones(row["batch_size"], row["seq_len"], dtype=torch.bool, device=device)
    return x, valid_mask


def run_series(
    row: dict,
    variant: str,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> dict:
    config = tb.TransformerConfig(
        batch_size=row["batch_size"],
        seq_len=row["seq_len"],
        d_model=row["d_model"],
        num_heads=row["heads"],
        ffn_dim=row["ffn_dim"],
        num_layers=row["layers"],
        causal=True,
    )
    config.validate()

    # Deterministic and distinct per (row, variant), same construction as the
    # harness's own args.seed + 100000 offset for generate_random_case.
    seed = 1234 + row["row"] * 10 + (0 if variant == "baseline" else 1)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_cls = tb.BaselineTransformer if variant == "baseline" else tb.UserOptimizedTransformer
    model = model_cls(config).to(device=device, dtype=dtype).eval()

    if row["row"] == ROW14:
        x, valid_mask = make_row14_input(row, device, seed=seed + 100000)
    else:
        x, valid_mask = tb.generate_random_case(
            config=config,
            device=device,
            dtype=dtype,
            seed=seed + 100000,
            padding_ratio=0.0,
            input_scale=1.0,
        )

    gpu_before = gpu_probe()
    tb.warmup_model(model, x, valid_mask, warmup, device)
    samples_ms = tb.benchmark_once(model, x, valid_mask, iterations, device)
    gpu_after = gpu_probe()

    mean_ms = statistics.fmean(samples_ms)
    sd_ms = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0

    print(
        f"row={row['row']:<2} variant={variant:<9} n={len(samples_ms):<4} "
        f"mean={mean_ms:.4f}ms sd={sd_ms:.4f}ms"
    )

    record = {
        "row": row["row"],
        "variant": variant,
        "shape": {k: v for k, v in row.items() if k != "row"},
        "n": len(samples_ms),
        "mean_ms": mean_ms,
        "sd_ms": sd_ms,
        "samples_ms": samples_ms,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }

    del model, x, valid_mask
    if device.type == "cuda":
        torch.cuda.empty_cache()
    time.sleep(2)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the official 14-row causal matrix, collecting raw "
        "per-iteration latency samples per (shape, variant) for distribution analysis."
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--row14-iterations",
        type=int,
        default=100,
        help="row 14's chunked optimized path over seq_len=100000 is far slower "
        "per call than the other 13 rows; default bounds its runtime",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"), default="high"
    )
    parser.add_argument(
        "--allow-tf32", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: results/artifacts/latency_distribution_<stamp>.json)",
    )
    parser.add_argument(
        "--only-row",
        type=int,
        default=None,
        help="sweep a single row number instead of the full matrix "
        "(row 14 must always be run this way, in its own process -- see module docstring)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = tb.resolve_device(args.device)
    dtype = tb.resolve_dtype(args.dtype)

    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(
        ROOT, "results", "artifacts", f"latency_distribution_{stamp}.json"
    )
    args.out = out_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print(f"device={device} dtype={dtype} torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    if args.only_row is None:
        rows = [r for r in OFFICIAL_ROWS if r["row"] != ROW14]
        print(
            f"skipping row {ROW14}: run it as its own process afterwards with "
            f"--only-row {ROW14} (see module docstring) -- a still-resident "
            "sibling process's CUDA context is enough to OOM it"
        )
    else:
        rows = [r for r in OFFICIAL_ROWS if r["row"] == args.only_row]

    series: List[dict] = []
    for row in rows:
        variants = ["optimized"] if row["row"] == ROW14 else ["baseline", "optimized"]
        iterations = args.row14_iterations if row["row"] == ROW14 else args.iterations
        for variant in variants:
            try:
                series.append(run_series(row, variant, device, dtype, args.warmup, iterations))
            except Exception as exc:  # keep sweeping remaining shapes on one failure
                print(f"row={row['row']} variant={variant} FAILED: {exc!r}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    payload = {
        "timestamp": stamp,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "row14_iterations": args.row14_iterations,
        "row14_note": (
            "row 14 is optimized-only; its model is fp32 like every other row, "
            "but its input tensor is built directly in fp16 (see "
            "make_row14_input() in tools/latency_distribution_sweep.py) since "
            "an fp32 input alone doesn't fit this card's memory for this shape"
        ),
        "git_commit": git_commit(),
        "series": series,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f)

    print(f"\nwrote {len(series)} series to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
