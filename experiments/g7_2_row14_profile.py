"""
G7.2 / Step 0 -- profile _chunked_forward_causal at official Row 14.

Decides where the roadmap's steps 1-7 actually pay off, before any is written.
No benchmark.py change.

Row 14 = B=32 d=1024 H=16 hd=64 S=100000 L=2 ffn=1024 causal => 49 chunks x 2
layers at chunk_q=2048. Baseline 13.0 s / 20.8 GB (job 198).

Roofline for the shape:
  attention   2*B*S^2*d*L            = 1.31e15 FLOP -> 7.9 s at 165 TFLOP/s
  the 8 GEMMs 2*B*S*d*(4d+2ffn)*L    = 8.05e13 FLOP -> 0.49 s

Method: torch.profiler -> chrome trace, then attribute every KERNEL to the CPU
op that launched it via `External id`. Two things make that necessary:
  * key_averages() double-counts -- the aten row and the kernel row each carry
    self_device_time, which inflates the total ~2x;
  * telling the two attention calls apart needs the launching op's input dims,
    and `_efficient_attention_forward` takes BSHD ([B, M, H, K]) -- the
    sequence length is dim 1, NOT the dim 2 that the BHSD-presenting
    _scaled_dot_product_efficient_attention would suggest.

sbatch only, via infra/slurm/g7_2_row14_profile.sbatch.
"""
import json
import os
import sys
import tempfile
import time
from collections import defaultdict

import torch

sys.path.insert(0, "/work")
import torch_transformer_benchmark as B  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROW14 = dict(bs=32, sl=100000, dm=1024, nh=16, ff=1024, nl=2, seed=1234)
PEAK_FP16_TFLOPS = 165.0     # RTX 4090 dense fp16-in/fp32-acc, CLAUDE.md table


def gb(n):
    return n / 1024 ** 3


def build(bs, sl, dm, nh, ff, nl):
    cfg = B.TransformerConfig(batch_size=bs, seq_len=sl, d_model=dm,
                              num_heads=nh, ffn_dim=ff, num_layers=nl,
                              causal=True)
    cfg.validate()
    torch.manual_seed(0)
    base = B.BaselineTransformer(cfg).to(DEV, torch.float32).eval()
    opt = B.UserOptimizedTransformer(cfg).to(DEV, torch.float32).eval()
    B.copy_model_weights(base, opt, strict=True)
    del base
    return cfg, opt


def make_input_fp16(bn, sn, dn, seed, tile=8192):
    x = torch.empty(bn, sn, dn, dtype=torch.float16, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for c0 in range(0, sn, tile):
        x[:, c0:min(c0 + tile, sn)].normal_(0.0, 1.0, generator=g)
    return x


def classify(op, dims, kname):
    """(phase, k_len) for one kernel, from its launching aten op."""
    n = (op or kname or "?").lower()
    if "efficient_attention" in n or "flash" in n or "fmha" in n:
        try:
            lq, lk = dims[0][1], dims[1][1]      # BSHD: seq is dim 1
            return ("SDPA.diag" if lq == lk else "SDPA.past"), lk
        except Exception:
            return "SDPA.?", None
    if "layer_norm" in n:
        return "layer_norm", None
    if "gelu" in n:
        return "gelu", None
    # bare "mm" also matches "command" (Command Buffer Full) -- match real names
    if any(t in n for t in ("addmm", "aten::mm", "aten::bmm", "aten::matmul",
                            "aten::linear", "gemm", "cutlass", "s1688",
                            "s16816", "ampere_", "tensorop")):
        return "GEMM", None
    if any(t in n for t in ("copy_", "to_copy", "contiguous", "clone",
                            "memcpy", "aten::cat")):
        return "cast/copy", None
    if any(t in n for t in ("exp", "maximum", "aten::add", "aten::mul",
                            "aten::div", "aten::sub", "elementwise", "fill",
                            "memset", "zero")):
        return "merge/elementwise", None
    return "other", None


def phases_from_trace(prof, iters):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    prof.export_chrome_trace(path)
    ev = json.load(open(path))["traceEvents"]
    os.unlink(path)

    ext2op = {}
    for e in ev:
        if (e.get("cat") or "") != "cpu_op":
            continue
        a = e.get("args") or {}
        ext = a.get("External id")
        if ext is not None:
            ext2op[ext] = (e.get("name"), a.get("Input Dims"))

    phase_us, phase_n = defaultdict(float), defaultdict(int)
    past_by_len, kern_us = defaultdict(float), defaultdict(float)
    n_kern = 0
    for e in ev:
        if (e.get("cat") or "").lower() not in ("kernel", "gpu_memcpy",
                                                "gpu_memset"):
            continue
        n_kern += 1
        dur = float(e.get("dur", 0.0))
        ext = (e.get("args") or {}).get("External id")
        opname, dims = ext2op.get(ext, (None, None))
        ph, klen = classify(opname, dims, e.get("name"))
        phase_us[ph] += dur
        phase_n[ph] += 1
        kern_us[e.get("name", "?")] += dur
        if ph == "SDPA.past" and klen is not None:
            past_by_len[klen] += dur
    return phase_us, phase_n, past_by_len, kern_us, n_kern


def profile_row14():
    from torch.profiler import ProfilerActivity, profile
    p = ROW14
    cfg, opt = build(p["bs"], p["sl"], p["dm"], p["nh"], p["ff"], p["nl"])
    mask = torch.ones(p["bs"], p["sl"], dtype=torch.bool, device=DEV)
    x = make_input_fp16(p["bs"], p["sl"], p["dm"], seed=p["seed"])

    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt(x, mask)
        torch.cuda.synchronize()
        warm_ms = (time.perf_counter() - t0) * 1e3
        print(f"  warmup forward: {warm_ms:.1f} ms", flush=True)
        iters = 2
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            for _ in range(iters):
                opt(x, mask)
            torch.cuda.synchronize()

    phase_us, phase_n, past_by_len, kern_us, n_kern = phases_from_trace(
        prof, iters)
    total = sum(phase_us.values())
    dev_ms = total / 1e3 / iters
    print(f"\n  total DEVICE time {dev_ms:.1f} ms/iter over {n_kern // iters:,}"
          f" kernels/iter   (wall {warm_ms:.0f} ms -> "
          f"{100 * dev_ms / warm_ms:.0f}% of wall is GPU-busy)")

    print(f"\n  {'phase':<20} {'ms/iter':>10} {'% device':>10} {'calls/iter':>11}")
    for ph, us in sorted(phase_us.items(), key=lambda kv: -kv[1]):
        print(f"  {ph:<20} {us / 1e3 / iters:10.1f} "
              f"{100 * us / total:9.1f}% {phase_n[ph] // iters:11,}")

    print(f"\n  top kernels by device time")
    for k, us in sorted(kern_us.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k[:60]:<60} {us / 1e3 / iters:9.1f}")

    if past_by_len:
        ks = sorted(past_by_len)
        print(f"\n  SDPA.past vs prefix length ({len(ks)} distinct k_len)")
        for klen in ks[:2] + ks[len(ks) // 2:len(ks) // 2 + 1] + ks[-2:]:
            print(f"    k_len={klen:>7,}  {past_by_len[klen] / 1e3 / iters:8.1f} ms")

    b, s_, d, ff, L = (p["bs"], p["sl"], p["dm"], p["ff"], p["nl"])
    attn_flops = 2.0 * b * s_ * s_ * d * L
    gemm_flops = 2.0 * b * s_ * d * (4 * d + 2 * ff) * L
    diag = phase_us.get("SDPA.diag", 0) / 1e3 / iters
    past = phase_us.get("SDPA.past", 0) / 1e3 / iters
    unk = phase_us.get("SDPA.?", 0) / 1e3 / iters
    sdpa_ms = diag + past + unk
    gemm_ms = phase_us.get("GEMM", 0) / 1e3 / iters
    ideal_attn = 1e3 * attn_flops / (PEAK_FP16_TFLOPS * 1e12)
    print(f"\n  -- roofline --")
    print(f"  attention {attn_flops:.3e} FLOP in {sdpa_ms:8.1f} ms -> "
          f"{attn_flops / max(sdpa_ms, 1e-9) * 1e3 / 1e12:7.1f} TFLOP/s "
          f"({100 * attn_flops / max(sdpa_ms, 1e-9) * 1e3 / 1e12 / PEAK_FP16_TFLOPS:.0f}%"
          f" of peak); ideal {ideal_attn:.0f} ms")
    print(f"  GEMMs     {gemm_flops:.3e} FLOP in {gemm_ms:8.1f} ms -> "
          f"{gemm_flops / max(gemm_ms, 1e-9) * 1e3 / 1e12:7.1f} TFLOP/s "
          f"({100 * gemm_flops / max(gemm_ms, 1e-9) * 1e3 / 1e12 / PEAK_FP16_TFLOPS:.0f}%"
          f" of peak); ideal {1e3 * gemm_flops / (PEAK_FP16_TFLOPS * 1e12):.0f} ms")
    other = dev_ms - sdpa_ms - gemm_ms
    print(f"\n  attention {100 * sdpa_ms / dev_ms:.0f}% of device time | "
          f"GEMMs {100 * gemm_ms / dev_ms:.0f}% | everything else {other:.0f} ms "
          f"({100 * other / dev_ms:.0f}%)")
    print(f"  SDPA.diag {diag:8.1f} ms ({phase_n['SDPA.diag'] // iters} calls)   "
          f"SDPA.past {past:8.1f} ms ({phase_n['SDPA.past'] // iters} calls)")
    print(f"  step 1/3 (flash) headroom inside attention: "
          f"{sdpa_ms - ideal_attn:.0f} ms to the roofline")
    print(f"  steps 4/6 (casts, LN, compile) contend for {other:.0f} ms total")

    with open("/work/results/g7_2_row14_profile_phases.json", "w") as fh:
        json.dump({"iters": iters, "warm_ms": warm_ms, "device_ms": dev_ms,
                   "phase_ms": {k: v / 1e3 / iters for k, v in phase_us.items()},
                   "phase_calls": {k: v // iters for k, v in phase_n.items()},
                   "kernel_ms": {k: v / 1e3 / iters for k, v in
                                 sorted(kern_us.items(),
                                        key=lambda kv: -kv[1])[:40]}}, fh, indent=1)
    del opt, x, mask
    torch.cuda.empty_cache()


def chunk_q_sweep():
    """Step 5's question asked early: does chunk_q matter at all?"""
    print("\n== chunk_q sweep (step 5, asked early) ==", flush=True)
    p = ROW14
    saved = B._CHUNK_Q
    print(f"  {'chunk_q':>8} {'n_chunks':>9} {'ms':>10} {'peak GB':>9}")
    try:
        for cq in (1024, 2048, 3072):
            try:
                B._CHUNK_Q = cq
                cfg, opt = build(p["bs"], p["sl"], p["dm"], p["nh"], p["ff"],
                                 p["nl"])
                mask = torch.ones(p["bs"], p["sl"], dtype=torch.bool, device=DEV)
                x = make_input_fp16(p["bs"], p["sl"], p["dm"], seed=p["seed"])
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    opt(x, mask)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    opt(x, mask)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) * 1e3
                peak = torch.cuda.max_memory_allocated()
                print(f"  {cq:>8} {(p['sl'] + cq - 1) // cq:>9} {ms:10.1f} "
                      f"{gb(peak):9.2f}", flush=True)
                del opt, x, mask, cfg
            except RuntimeError as e:
                print(f"  {cq:>8} {'':>9} {'OOM/err':>10}  {str(e)[:60]}")
            finally:
                torch.cuda.empty_cache()
    finally:
        B._CHUNK_Q = saved


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM {gb(free):.2f}/{gb(total):.2f} GB  "
          f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")
    print("\n== Step 0: kernel census, Row 14 ==", flush=True)
    profile_row14()
    chunk_q_sweep()
    print("\nG7_2_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
