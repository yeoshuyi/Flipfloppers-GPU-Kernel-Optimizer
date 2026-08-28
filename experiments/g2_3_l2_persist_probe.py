#!/usr/bin/env python3
"""
G2.3 probe -- does a CUDA L2 accessPolicyWindow actually help this model's
weight-streaming pattern, and does it survive torch.compile's CUDA-graph
capture (which records on its OWN stream, not the one we would normally set
the attribute on)?

Answering the second question BEFORE trusting any benchmark.py number matters:
the shipped path is torch.compile(mode="reduce-overhead"), so a window that
only applies to eagerly-launched kernels is worth nothing there.

Four configurations per shape, identical work in each:
  A eager,  no window
  B eager,  window set on the current (default) stream
  C graph,  no window on the capture stream
  D graph,  window set INSIDE the capture region (so it lands on the capture
            stream and is snapshotted into the graph's kernel nodes)

Swept over TINY / DEFAULT / LARGE-BATCH, because the catalogue's claim is
regime-specific ("1.1x default, 1.5x+ tiny") -- tiny is the shape where the
~60MB of weights dominate the per-forward byte traffic.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "csrc", "l2_persist.cpp")

D, FF, L = 512, 2048, 6


def build_ext():
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    # with_cuda=True is REQUIRED even though there is no .cu source: torch
    # auto-detects CUDA only from .cu files, so without it the CUDA include
    # path and -lcudart are both missing and <cuda_runtime.h> is not found.
    return load(name="l2_persist", sources=[SRC], build_directory=build_dir,
                with_cuda=True, verbose=False)


def make_arena(dev):
    specs = [[((3 * D, D), torch.float16), ((D, D), torch.float16),
              ((FF, D), torch.float32), ((D, FF), torch.float32)]
             for _ in range(L)]
    align = 256
    total = 0
    for grp in specs:
        for shape, dt in grp:
            n = shape[0] * shape[1] * torch.empty((), dtype=dt).element_size()
            total += (n + align - 1) // align * align
    arena = torch.empty(total, dtype=torch.uint8, device=dev)
    layers, off = [], 0
    for grp in specs:
        ws = []
        for shape, dt in grp:
            n = shape[0] * shape[1] * torch.empty((), dtype=dt).element_size()
            v = arena.narrow(0, off, n).view(dt).view(shape)
            v.normal_(0, 0.02)
            ws.append(v)
            off += (n + align - 1) // align * align
        layers.append(ws)
    return arena, layers, total


def timeit(fn, iters=300, warm=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def capture(step, x0):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step(x0)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return g


def main() -> int:
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    ext = build_ext()

    l2 = ext.l2_cache_size()
    pmax = ext.persisting_l2_max_size()
    wmax = ext.access_policy_max_window_size()
    print("=== device L2 facts ===")
    print(f"l2CacheSize                = {l2} ({l2/2**20:.1f} MiB)")
    print(f"persistingL2CacheMaxSize   = {pmax} ({pmax/2**20:.1f} MiB)")
    print(f"accessPolicyMaxWindowSize  = {wmax} ({wmax/2**20:.1f} MiB)")

    arena, layers, nbytes = make_arena(dev)
    granted = ext.set_persist_limit(min(nbytes, pmax))
    ratio = min(1.0, granted / float(nbytes))
    print(f"arena                      = {nbytes} ({nbytes/2**20:.1f} MiB)")
    print(f"granted persisting limit   = {granted} ({granted/2**20:.1f} MiB)")
    print(f"hit_ratio                  = {ratio:.4f}", flush=True)

    def step(x):
        for qkv_w, op_w, fi_w, fo_w in layers:
            h = F.linear(x.to(torch.float16), qkv_w)
            h = F.linear(h[..., :D], op_w).to(torch.float32)
            x = x + h
            g = F.gelu(F.linear(x, fi_w))
            x = x + F.linear(g, fo_w)
        return x

    for (B, S), label in [((1, 64), "TINY"), ((8, 128), "DEFAULT"),
                          ((256, 128), "LARGE-BATCH")]:
        x0 = torch.randn(B, S, D, device=dev)
        print(f"\n=== {label}  B={B} S={S} ===", flush=True)

        ext.reset_window()
        ext.reset_persisting_l2()
        a = timeit(lambda: step(x0))
        ext.set_window(arena, ratio)
        b = timeit(lambda: step(x0))
        ext.reset_window()
        ext.reset_persisting_l2()

        gC = capture(step, x0)
        with torch.cuda.graph(gC):
            step(x0)
        c = timeit(gC.replay)
        del gC

        gD = capture(step, x0)
        try:
            with torch.cuda.graph(gD):
                ext.set_window(arena, ratio)
                step(x0)
            d = timeit(gD.replay)
            dtxt = f"{d:8.4f}   x{c/d:.4f} vs C"
        except Exception as exc:  # noqa: BLE001
            d = None
            dtxt = f"FAILED {type(exc).__name__}: {exc}"
        del gD
        ext.reset_window()
        ext.reset_persisting_l2()

        gbps = nbytes / (a / 1e3) / 1e9
        print(f"A eager,  no window       {a:8.4f}   "
              f"(weight traffic {gbps:6.1f} GB/s if fetched from DRAM)")
        print(f"B eager,  window          {b:8.4f}   x{a/b:.4f} vs A")
        print(f"C graph,  no window       {c:8.4f}")
        print(f"D graph,  window in capt  {dtxt}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
