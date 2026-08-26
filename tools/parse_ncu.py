#!/usr/bin/env python3
"""
Parses `ncu --metrics ... --csv` output (read from stdin) into the compact
JSON fact block .claude/agents/profiler.md's contract expects. This is the
piece described but never built in docs/SETUP.md ("Parse ncu to JSON in
tools/ -- never let raw output reach an agent's context") -- raw ncu CSV
for a real run is large; this script is meant to sit between `ncu` and any
agent/human, emitting ONLY the small JSON below on stdout.

Usage:
    ncu --metrics <list> --csv --target-processes all <cmd> \
        | python3 tools/parse_ncu.py [--dtype tf32|bf16|fp8]

Format notes (discovered empirically, jobs/ncu_header_check2.sbatch):
  - ncu's own "==PROF== Connected..." banner AND the profiled program's own
    stdout prints land on the SAME stream, BEFORE the CSV -- this script
    skips everything before the real header line.
  - The CSV is LONG format: one row per (kernel launch ID, metric), with
    "ID","Kernel Name",...,"Metric Name","Metric Value" columns -- NOT one
    row per kernel with a metric-per-column. Rows are grouped by "ID".

Metric name discovery (jobs/ncu_discover.sbatch, jobs/ncu_discover2.sbatch)
found no launch__* metrics (registers-per-thread, shared-mem-per-block) in
this ncu build/version -- regs_per_thread, shared_bytes_per_block, and
reg_spills are reported as null with a note rather than guessed.
"""
import csv
import io
import json
import sys
import argparse

PEAK_TFLOPS = {"tf32": 82.6, "bf16": 165.2, "fp8": 330.3}
# RTX 4090, GDDR6X, per NVIDIA's published spec -- used only to convert
# dram__throughput's pct_of_peak into an approximate GB/s figure.
PEAK_DRAM_GBPS = 1008.0

STALL_METRICS = {
    "long_scoreboard": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active",
    "barrier": "smsp__warp_issue_stalled_barrier_per_warp_active",
    "mio_throttle": "smsp__warp_issue_stalled_mio_throttle_per_warp_active",
    "short_scoreboard": "smsp__warp_issue_stalled_short_scoreboard_per_warp_active",
    "not_selected": "smsp__warp_issue_stalled_not_selected_per_warp_active",
    "wait": "smsp__warp_issue_stalled_wait_per_warp_active",
}

DURATION_KEY = "gpu__time_duration.sum"
SM_THROUGHPUT_KEY = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
DRAM_THROUGHPUT_KEY = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
LTS_THROUGHPUT_KEY = "lts__throughput.avg.pct_of_peak_sustained_elapsed"
OCCUPANCY_KEY = "sm__warps_active.avg.pct_of_peak_sustained_active"
BANK_CONFLICT_KEY = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"
TENSOR_INST_KEY = "sm__inst_executed_pipe_tensor_op_hmma_v2.sum"


def _to_float(v: str) -> float | None:
    if v is None:
        return None
    v = v.strip().replace(",", "")
    if v in ("", "n/a", "N/A"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _extract_csv(raw: str) -> str:
    # Skip ncu's banner + the profiled program's own stdout prints, which
    # land on the same stream before the real CSV.
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('"ID"') and "Metric Name" in line:
            return "\n".join(lines[i:])
    return ""


def parse(raw: str, dtype: str) -> dict:
    csv_text = _extract_csv(raw)
    if not csv_text:
        return {"error": "no CSV header ('\"ID\"...\"Metric Name\"...') found in ncu output"}

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return {"error": "CSV header found but no data rows parsed"}

    # Group by kernel-launch ID: each launch contributes one row per
    # requested metric (long format).
    launches: dict[str, dict] = {}
    for row in rows:
        launch_id = row.get("ID")
        if launch_id is None:
            continue
        entry = launches.setdefault(launch_id, {"name": row.get("Kernel Name", "?"), "metrics": {}})
        metric_name = row.get("Metric Name")
        metric_value = _to_float(row.get("Metric Value"))
        if metric_name is not None:
            entry["metrics"][metric_name] = metric_value

    per_kernel = []
    total_dur_ns = 0.0
    for entry in launches.values():
        m = entry["metrics"]
        dur = m.get(DURATION_KEY) or 0.0
        total_dur_ns += dur
        per_kernel.append({
            "name": entry["name"],
            "dur_ns": dur,
            "sm_pct": m.get(SM_THROUGHPUT_KEY),
            "dram_pct": m.get(DRAM_THROUGHPUT_KEY),
            "lts_pct": m.get(LTS_THROUGHPUT_KEY),
            "occ_pct": m.get(OCCUPANCY_KEY),
            "bank_conflicts": m.get(BANK_CONFLICT_KEY),
            "tensor_insts": m.get(TENSOR_INST_KEY),
            "stalls": {name: m.get(key) for name, key in STALL_METRICS.items()},
        })

    per_kernel.sort(key=lambda k: k["dur_ns"], reverse=True)
    top3 = per_kernel[:3]
    hot_kernels = [
        {
            # Kernel names here are template-mangled C++ (very long); trim
            # for readability in the compact fact block.
            "name": (k["name"][:120] + "...") if len(k["name"]) > 120 else k["name"],
            "us": round(k["dur_ns"] / 1000.0, 3),
            "pct_of_total": round(100.0 * k["dur_ns"] / total_dur_ns, 2) if total_dur_ns else None,
        }
        for k in top3
    ]

    def weighted(field: str) -> float | None:
        if not total_dur_ns:
            return None
        acc = sum((k[field] or 0.0) * k["dur_ns"] for k in per_kernel)
        return acc / total_dur_ns

    weighted_sm_pct = weighted("sm_pct")
    weighted_dram_pct = weighted("dram_pct")
    weighted_occ_pct = weighted("occ_pct")

    peak = PEAK_TFLOPS.get(dtype)
    achieved_tflops = (weighted_sm_pct / 100.0 * peak) if (weighted_sm_pct is not None and peak) else None
    dram_gbps = (weighted_dram_pct / 100.0 * PEAK_DRAM_GBPS) if weighted_dram_pct is not None else None

    stall_avgs = {}
    for name in STALL_METRICS:
        vals = [(k["stalls"][name], k["dur_ns"]) for k in per_kernel if k["stalls"][name] is not None]
        if vals and total_dur_ns:
            stall_avgs[name] = sum(v * d for v, d in vals) / total_dur_ns
    top_stall, stall_pct = (None, None)
    if stall_avgs:
        top_stall = max(stall_avgs, key=stall_avgs.get)
        stall_pct = round(stall_avgs[top_stall], 2)

    total_bank_conflicts = sum(k["bank_conflicts"] or 0.0 for k in per_kernel)
    graph_replay = any("cudagraph" in k["name"].lower() or "graph" in k["name"].lower() for k in per_kernel)

    return {
        "hot_kernels": hot_kernels,
        "achieved_tflops": round(achieved_tflops, 2) if achieved_tflops is not None else None,
        "pct_of_peak": round(weighted_sm_pct, 2) if weighted_sm_pct is not None else None,
        "pct_of_peak_note": f"sm__throughput (overall SM), not tensor-core-isolated; dtype={dtype}, peak={peak}",
        "dram_gbps": round(dram_gbps, 2) if dram_gbps is not None else None,
        "l2_gbps": None,
        "l2_gbps_note": "no reliable peak L2 bandwidth figure available; see lts_pct per-kernel instead",
        "occupancy_pct": round(weighted_occ_pct, 2) if weighted_occ_pct is not None else None,
        "regs_per_thread": None,
        "shared_bytes_per_block": None,
        "reg_spills": None,
        "launch_metrics_note": "no launch__* metrics found in this ncu build (jobs/ncu_discover2.sbatch) -- regs/shared-mem/spills not available",
        "bank_conflicts": int(total_bank_conflicts),
        "top_stall": top_stall,
        "stall_pct": stall_pct,
        "kernel_launches": len(per_kernel),
        "gpu_idle_pct": None,
        "graph_replay": graph_replay,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="tf32", choices=list(PEAK_TFLOPS))
    a = ap.parse_args()
    raw = sys.stdin.read()
    print(json.dumps(parse(raw, a.dtype), indent=2))
