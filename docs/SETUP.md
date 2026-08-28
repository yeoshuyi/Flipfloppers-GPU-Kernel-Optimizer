# Setup — Phase 0, Slurm, Apptainer, Measurement Protocol

Do all of this **before** the first optimisation. Everything downstream depends
on facts measured on *this* machine and on timings that are not noise.

**Host:** `ubuntu-makers` — Ryzen 7 7700, 32 GiB DDR5, RTX 4090 24 GB (sm_89),
Ubuntu 26.04, CUDA 13.1, Tailscale-only access.

---

## 1. Topology

Run Claude Code **on the server, inside `tmux`**. The agent needs tight
filesystem and `sbatch` access, and `tmux` survives Tailscale reconnects.

```bash
ssh ubuntu-makers
tmux new -s kernel-agent          # or: tmux attach -t kernel-agent

# once
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
npm install -g @anthropic-ai/claude-code
claude setup-token                # long-lived token for headless runs
```

---

## 2. Why Slurm on a single-GPU box

**Not for scheduling — for measurement hygiene.** Two concurrent benchmark
candidates contend for SMs, clocks, and power budget, and every number produced
is noise. `--exclusive --gres=gpu:1` serialises GPU access and makes timings
reproducible. Free job accounting for the report is a bonus.

```bash
# /etc/slurm/slurm.conf
NodeName=ubuntu-makers CPUs=16 RealMemory=30000 Gres=gpu:rtx4090:1 State=UNKNOWN
PartitionName=gpu Nodes=ubuntu-makers Default=YES MaxTime=02:00:00 State=UP
PrologFlags=Alloc

# /etc/slurm/gres.conf
NodeName=ubuntu-makers Name=gpu Type=rtx4090 File=/dev/nvidia0
```

### Clock locking belongs in the prolog, not in the agent

Unlocked clocks invalidate every measurement in this project. Enforce it where
the agent cannot reach it.

```bash
# /etc/slurm/prolog.sh
#!/bin/bash
nvidia-smi -pm 1
nvidia-smi -lgc 2520        # pin SM clock; tune to your stable boost
nvidia-smi -pl 450

# /etc/slurm/epilog.sh
#!/bin/bash
nvidia-smi -rgc
```

`nvidia-smi -lgc` / `-pl` / `-r` are in the agent's **deny** list. An agent
resetting clocks mid-sweep silently corrupts everything measured afterward.

---

## 3. Apptainer

Rootless, GPU passthrough via `--nv`, single-file images on `/scratch`.

```
Bootstrap: docker
From: nvidia/cuda:13.1.0-devel-ubuntu24.04

%post
    apt-get update && apt-get install -y python3-pip git cmake ninja-build
    pip3 install --break-system-packages \
        torch --index-url https://download.pytorch.org/whl/cu131
    pip3 install --break-system-packages triton numpy pandas
    apt-get install -y cuda-nsight-compute-13-1 cuda-nsight-systems-13-1

%environment
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export TORCH_CUDA_ARCH_LIST="8.9"      # sm_89 only -- much faster JIT builds
    export CUDA_MODULE_LOADING=LAZY

%runscript
    exec python3 "$@"
```

```bash
apptainer build /scratch/kernel.sif infra/apptainer/kernel.def

# ncu needs perf counter access, otherwise every profile returns empty
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
    > /etc/modprobe.d/nvidia-profiler.conf'
sudo update-initramfs -u && sudo reboot
```

---

## 4. Job wrapper

```bash
#!/bin/bash
#SBATCH --job-name=kbench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/techjam2/runs/%j.out

set -euo pipefail
CANDIDATE="$1"; SHAPE_CFG="$2"; MODE="${3:-bench}"

apptainer exec --nv --cleanenv \
  --bind /scratch/work:/work --bind /scratch/techjam2/runs:/runs \
  /scratch/kernel.sif \
  python3 /work/harness/run.py \
      --candidate "$CANDIDATE" --shape "$SHAPE_CFG" --mode "$MODE" \
      --out "/runs/$SLURM_JOB_ID.json"
```

## 5. Tool layer — submit, release, poll

**The agent must never hold the GPU while thinking.** A blocking `srun` inside
an agent turn wastes the GPU for the duration of model inference.

```python
import subprocess, json, time, pathlib
RUNS = pathlib.Path("/scratch/techjam2/runs")

def submit(candidate, shape, mode="bench"):
    out = subprocess.run(
        ["sbatch", "--parsable", "/scratch/work/jobs/bench.sbatch",
         candidate, shape, mode],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()

def poll(job_id, timeout_s=1500):
    result = RUNS / f"{job_id}.json"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if result.exists():
            return json.loads(result.read_text())
        state = subprocess.run(["sacct", "-j", job_id, "-n", "-o", "State"],
                               capture_output=True, text=True).stdout
        if any(s in state for s in ("FAILED", "TIMEOUT", "CANCELLED")):
            return {"error": state.strip(),
                    "log": (RUNS / f"{job_id}.out").read_text()[-4000:]}
        time.sleep(5)
    subprocess.run(["scancel", job_id])
    return {"error": "timeout"}
```

---

## 6. Phase 0 capability probe

**Assume nothing.** Run via `sbatch`, record every answer, and treat the output
as the denominator of every claim you later make.

```python
#!/usr/bin/env python3
"""Phase 0: establish ground truth on THIS machine."""
import torch, json

r = {"torch": torch.__version__, "cuda": torch.version.cuda,
     "cc": torch.cuda.get_device_capability(), "gpu": torch.cuda.get_device_name(0)}

# --- FP8 library path (gates the whole G4/FP8 plan) --------------------
try:
    a = torch.randn(64, 128, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(128, 64, device="cuda").to(torch.float8_e4m3fn).t()
    s = torch.ones(1, device="cuda", dtype=torch.float32)
    torch._scaled_mm(a, b, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
    r["fp8_scaled_mm"] = True
except Exception as e:
    r["fp8_scaled_mm"] = f"NO: {type(e).__name__}: {e}"

# --- Triton FP8 cast (needs PTX ISA >= 8.1 on sm_89) -------------------
try:
    import triton, triton.language as tl
    @triton.jit
    def _k(o, i, n, BS: tl.constexpr):
        off = tl.program_id(0) * BS + tl.arange(0, BS)
        tl.store(o + off, tl.load(i + off, mask=off < n).to(tl.float8e4nv),
                 mask=off < n)
    x = torch.randn(1024, device="cuda")
    y = torch.empty(1024, device="cuda", dtype=torch.float8_e4m3fn)
    _k[(1,)](y, x, 1024, BS=1024)
    r["triton_fp8"] = True
except Exception as e:
    r["triton_fp8"] = f"NO: {type(e).__name__}: {e}"

# --- cooperative launch (gates G4.1) -----------------------------------
r["cooperative_launch"] = bool(torch.cuda.get_device_properties(0).__dict__
                               .get("cooperative_launch", True))

# --- hardware limits ----------------------------------------------------
p = torch.cuda.get_device_properties(0)
r.update(sm_count=p.multi_processor_count,
         shared_per_sm_kb=p.shared_memory_per_multiprocessor / 1024,
         l2_mb=p.L2_cache_size / 1e6,
         regs_per_sm=p.regs_per_multiprocessor)

# --- achieved peak per dtype -------------------------------------------
def peak(dtype, n=8192, iters=50):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(10): a @ b
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): a @ b
    e.record(); torch.cuda.synchronize()
    return 2 * n**3 / (s.elapsed_time(e) / iters * 1e-3) / 1e12

torch.backends.cuda.matmul.allow_tf32 = True
r["tf32_tflops"] = peak(torch.float32)
r["bf16_tflops"] = peak(torch.bfloat16)
r["fp16_tflops"] = peak(torch.float16)

print(json.dumps(r, indent=2))
```

### Reading the result

| Probe | If NO |
|---|---|
| `fp8_scaled_mm` | FP8 goes via Triton or CUTLASS, not the library path |
| `triton_fp8` | FP8 goes via CUDA C++ / raw PTX |
| `cooperative_launch` | G4.1 is out; G4.0 two-kernel form is the ceiling |
| `bf16_tflops` ≈ 80 not ≈120–160 | **Power or clock throttled.** Fix before anything else — every later measurement inherits the error. |

Expected: `tf32` ≈ 60–80, `bf16`/`fp16` ≈ 120–160.

### Then measure the baseline

Full disclosed shape sweep. Record median, p90, min, and `nsys` launch count per
shape. **This table is the denominator of every claim in the report.**

---

## 7. Measurement protocol

### Anti-noise
- Clocks locked in the prolog; `--exclusive` on every job.
- Report **median** and p90. p90 exposes CUDA Graph replay jitter.
- Log `clocks.sm` and `temperature.gpu` alongside every run. A 450 W part in a
  desktop chassis **will** drift; if SM clock falls >5% during a sweep, insert
  cooldowns.
- `torch.cuda.empty_cache()` plus a 2 s settle between configs.
- The harness already alternates baseline/optimised order across rounds — keep
  `--benchmark-rounds ≥ 3`.

### Accuracy
```
external  max_abs 1e-3 OR max_rel 1e-2   (disjunctive, per element)
internal  max_abs 5e-4                   <- investigate above this
```
- Validate **per shape**, never once.
- Track the *trend*: 3e-4 → 8e-4 is a warning even though both pass.
- Test `--padding-ratio 0.3` and `--causal`; both change the code path.
- After every transformation claimed exact (G1.x), **verify `max_abs` is
  bit-identical**. If folding moved the error, the fold is wrong.

### Profiling
```bash
ncu --set full --target-processes all \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section WarpStateStats --section Occupancy \
    -o prof_$(date +%s) python benchmark.py [args]

nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
     -o timeline_$(date +%s) python benchmark.py [args]
```
Wrap logical stages in `torch.cuda.nvtx.range_push/pop` so the timeline is
readable. **Parse `ncu` to JSON in `tools/` — never let raw output reach an
agent's context.**

---

## 8. Directory layout

```
/scratch/work/
  CLAUDE.md                 always-loaded project memory
  .claude/
    settings.json           permissions
    agents/profiler.md      the one subagent
  docs/                     load-on-demand references
  benchmark.py              unmodified reference harness
  src/
    model.py                UserOptimizedTransformer + shape dispatch
    precompute.py           G1 -- fold, swizzle, quantise, arena
    kernels/{triton,cuda,ptx}/
  harness/run.py            invoked inside Apptainer by the sbatch wrapper
  tools/                    check_validity.py, archive.py, submit/poll
  jobs/bench.sbatch
  probes/phase0.py
  archive/                  MAP-Elites cells + lineage
/scratch/techjam2/runs/              job outputs: one JSON + one .out per job id
/scratch/kernel.sif         Apptainer image
```

---

## 9. Schedule

| Day | Deliverable | Gate |
|---|---|---|
| 1 | Phase 0 probe, clock lock, baseline sweep | FP8 availability known; baseline table recorded |
| 1–2 | **Track A** complete and verified | A passing submission exists. **Non-negotiable.** |
| 2–3 | Slurm, Apptainer, tool layer, `CLAUDE.md` | `submit()`/`poll()` round-trips cleanly |
| 3–5 | G1 precompute | **`max_abs` unchanged** — folding must be exact |
| 5–8 | G4.0 two-kernel form | Passes sweep; beats Track A |
| 8–10 | G3 fusion + layout (agent-driven) | — |
| 10–12 | FP8 FFN, per-channel scales | Holds on **all** shapes, or fall back to split-precision |
| 12–16 | G4.1–G4.3 megakernel | Correctness first, performance second |
| 16–18 | Solidification — strip autotune, freeze constants | No runtime branches in the submission |
| 18–21 | Full sweep, ablation, report | — |

**Track A** is the safety net: G0 + G1 + G2 only (SDPA, fused QKV, killed
transposes, constant folding, BF16, L2 pin, CUDA Graphs). One to two days,
entirely conventional, ~5–8×. It exists so that a megakernel that does not
converge by the deadline still leaves you with a submission.
