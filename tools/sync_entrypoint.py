#!/usr/bin/env python3
"""
Generate a standalone torch_transformer_benchmark.py = the judges' canonical
baseline script + our UserOptimizedTransformer spliced in.

The grader's invocation contract is unconfirmed (does it run our benchmark.py,
splice our class into its own torch_transformer_benchmark.py, or import our
class?). This produces the safe superset: a single self-contained file that is
the judges' harness verbatim with our optimized model dropped in, so
`python3 torch_transformer_benchmark.py ...` behaves exactly like
`python3 benchmark.py ...`.

    python3 tools/sync_entrypoint.py
    python3 tools/sync_entrypoint.py --check     # exit 1 if the file is stale

Re-run after ANY change to benchmark.py or ~/torch_transformer_benchmark.py.
tools/verify_baseline.py guards the harness half; this keeps the model half in
sync.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OURS = os.path.join(ROOT, "benchmark.py")
CANON = os.path.expanduser("~/torch_transformer_benchmark.py")
OUT = os.path.join(ROOT, "torch_transformer_benchmark.py")

HEADER = (
    "# TikTok TechJam - Problem 3 submission entry point.\n"
    "#\n"
    "# This is the official torch_transformer_benchmark.py with "
    "UserOptimizedTransformer\n"
    "# (and its module-level helpers) implemented in place of the stub. The "
    "scoring\n"
    "# half - BaselineTransformer, compare_outputs, run_accuracy_tests, "
    "benchmark_models,\n"
    "# parse_args, main - is byte-for-byte the reference harness; "
    "tools/verify_baseline.py\n"
    "# asserts that on every build.\n"
    "#\n"
    "#   python3 torch_transformer_benchmark.py --causal --batch-size <B> "
    "--seq-len <S> \\\n"
    "#       --d-model <d> --heads <H> --layers <L> --ffn-dim <F>\n"
    "#\n"
    "# Auto-generated from benchmark.py by tools/sync_entrypoint.py - edit "
    "benchmark.py,\n"
    "# not this file, then re-run tools/sync_entrypoint.py.\n\n"
)


def _idx(lines, prefix, start=0):
    for i in range(start, len(lines)):
        if lines[i].startswith(prefix):
            return i
    raise SystemExit(f"sync_entrypoint: marker not found: {prefix!r}")


def build() -> str:
    ours = open(OURS).read().splitlines(keepends=True)
    canon = open(CANON).read().splitlines(keepends=True)

    # --- our module-level helpers: last import .. class TransformerConfig ---
    o_imp = _idx(ours, "import torch.nn.functional as F")
    o_cfg = _idx(ours, "class TransformerConfig")
    hlines = ours[o_imp + 1:o_cfg]
    # drop the decorator/blank lines that belong to `class TransformerConfig`
    # (which itself comes from the canonical harness, not from us)
    while hlines and (hlines[-1].strip() == "" or hlines[-1].lstrip().startswith("@")):
        hlines.pop()
    helpers = "".join(hlines).strip("\n") + "\n"

    # --- our optimized model: class UserOptimizedTransformer .. next def ----
    o_cls = _idx(ours, "class UserOptimizedTransformer")
    o_end = _idx(ours, "def copy_model_weights", o_cls)
    model = "".join(ours[o_cls:o_end]).strip("\n") + "\n"

    # --- canonical harness, with our block replacing its stub class --------
    c_cls = _idx(canon, "class UserOptimizedTransformer")
    c_end = _idx(canon, "def copy_model_weights", c_cls)
    head = canon[:c_cls]
    tail = canon[c_end:]

    # our helpers use os.environ; the canonical imports do not include os
    for i, ln in enumerate(head):
        if ln.startswith("import math"):
            if not any(x.startswith("import os") for x in head):
                head.insert(i + 1, "import os\n")
            break

    body = (
        "".join(head).rstrip("\n")
        + "\n\n\n" + helpers
        + "\n\n" + model
        + "\n\n\n" + "".join(tail).lstrip("\n")
    )
    # keep a shebang on line 1 if the canonical had one; header goes under it
    if body.startswith("#!"):
        nl = body.index("\n") + 1
        return body[:nl] + HEADER + body[nl:]
    return HEADER + body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if torch_transformer_benchmark.py is stale")
    a = ap.parse_args()

    if not os.path.exists(CANON):
        print(f"sync_entrypoint: canonical not found: {CANON}")
        return 1

    new = build()
    try:
        import ast
        ast.parse(new)
    except SyntaxError as e:
        print(f"sync_entrypoint: generated file does not parse: {e}")
        return 1

    if a.check:
        cur = open(OUT).read() if os.path.exists(OUT) else ""
        if cur != new:
            print("sync_entrypoint: torch_transformer_benchmark.py is STALE "
                  "-- run `python3 tools/sync_entrypoint.py`")
            return 1
        print("sync_entrypoint: torch_transformer_benchmark.py is up to date")
        return 0

    open(OUT, "w").write(new)
    n = new.count("\n")
    print(f"sync_entrypoint: wrote {OUT} ({n} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
