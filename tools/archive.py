#!/usr/bin/env python3
"""
MAP-Elites archive. Replaces the 'archivist' agent -- this is file I/O,
not a reasoning task. Zero tokens.

    python3 tools/archive.py commit  --cell default/fp8 --id cand_0042 \
                                     --speedup 12.4 --applied G0.1,G1.1
    python3 tools/archive.py query   --cell default/fp8
    python3 tools/archive.py fail    --cell default/fp8 --id cand_0043 \
                                     --reason "abs drift at S=1024"
    python3 tools/archive.py summary
"""
import json, sys, argparse, pathlib, subprocess, datetime

ROOT = pathlib.Path("archive")
REGIMES = ["tiny", "default", "long-seq", "large-batch", "padded", "causal"]
FAMILIES = ["trackA", "precompute", "two-kernel", "fp8", "megakernel"]


def _p(cell): return ROOT / f"{cell.replace('/', '__')}.json"


def _load(cell):
    f = _p(cell)
    return json.loads(f.read_text()) if f.exists() else {"elite": None, "log": []}


def _save(cell, d):
    ROOT.mkdir(exist_ok=True)
    _p(cell).write_text(json.dumps(d, indent=2))


def commit(a):
    d = _load(a.cell)
    entry = {"id": a.id, "speedup": a.speedup,
             "applied": a.applied.split(",") if a.applied else [],
             "ts": datetime.datetime.now().isoformat(timespec="seconds")}
    prev = d["elite"]["speedup"] if d["elite"] else 0
    if a.speedup > prev:
        d["elite"] = entry
        print(f"NEW ELITE {a.cell}: {a.speedup:.2f}x (was {prev:.2f}x)")
        subprocess.run(["git", "add", "-A"], check=False)
        subprocess.run(["git", "commit", "-m",
                        f"[{a.cell}] {a.id} {a.speedup:.2f}x {a.applied or ''}"],
                       check=False)
    else:
        print(f"near-miss {a.cell}: {a.speedup:.2f}x vs elite {prev:.2f}x")
    d["log"].append(entry)
    _save(a.cell, d)


def fail(a):
    d = _load(a.cell)
    d["log"].append({"id": a.id, "FAILED": a.reason,
                     "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    _save(a.cell, d)
    print(f"logged failure in {a.cell}: {a.reason}")


def query(a):
    d = _load(a.cell)
    print(json.dumps({"elite": d["elite"],
                      "failures": [e for e in d["log"] if "FAILED" in e][-5:]},
                     indent=2))


def summary(_):
    print(f"{'':<14}" + "".join(f"{f:>13}" for f in FAMILIES))
    for r in REGIMES:
        row = f"{r:<14}"
        for f in FAMILIES:
            e = _load(f"{r}/{f}")["elite"]
            cell = "{:.2f}x".format(e["speedup"]) if e else "-"
            row += "{:>13}".format(cell)
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("commit", commit), ("fail", fail), ("query", query), ("summary", summary)]:
        p = sub.add_parser(name); p.set_defaults(fn=fn)
        if name != "summary": p.add_argument("--cell", required=True)
        if name in ("commit", "fail"): p.add_argument("--id", required=True)
        if name == "commit":
            p.add_argument("--speedup", type=float, required=True)
            p.add_argument("--applied", default="")
        if name == "fail": p.add_argument("--reason", required=True)
    a = ap.parse_args(); a.fn(a)
