#!/usr/bin/env python3
"""
Guard against drift between our benchmark.py's FROZEN half (the reference model
+ the scoring harness) and the judges' canonical baseline script.

The judges publish `torch_transformer_benchmark.py`; our benchmark.py must carry
a byte-equivalent copy of everything except `UserOptimizedTransformer` and its
module-level helpers. If the judges update the harness (as they did 2026-08-27,
loosening atol/rtol 0.001/0.01 -> 0.002/0.02), this catches it before a ship.

    python3 tools/verify_baseline.py
    python3 tools/verify_baseline.py --ours benchmark.py --canonical ~/torch_transformer_benchmark.py

Exit 1 on any mismatch. Comparison is AST-level (ast.dump), so comments,
formatting, blank lines and line numbers do not matter -- only semantics.
"""
import argparse
import ast
import os
import sys

# Every top-level symbol that must stay identical to the judges' script.
# UserOptimizedTransformer and the _lt_/_ws_/_ffn_ helpers are OURS and are
# deliberately excluded.
FROZEN = [
    "TransformerConfig",
    "BaselineSelfAttention",
    "BaselineTransformerBlock",
    "BaselineTransformer",
    "copy_model_weights",
    "resolve_device",
    "resolve_dtype",
    "generate_random_case",
    "AccuracyResult",
    "compare_outputs",
    "run_accuracy_tests",
    "percentile",
    "TimingResult",
    "warmup_model",
    "benchmark_once",
    "benchmark_models",
    "maybe_compile",
    "parse_args",
    "validate_args",
    "main",
]

EXPECT_DEFAULTS = {"atol": 0.002, "rtol": 0.02}


def top_level_defs(path):
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
    return out


def arg_defaults(node):
    """{--flag: literal default} for a parse_args() function def."""
    found = {}
    for sub in ast.walk(node):
        if not (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "add_argument"):
            continue
        if not sub.args or not isinstance(sub.args[0], ast.Constant):
            continue
        flag = sub.args[0].value.lstrip("-").replace("-", "_")
        for kw in sub.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                found[flag] = kw.value.value
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmark.py"))
    ap.add_argument("--canonical", default=os.path.expanduser(
        "~/torch_transformer_benchmark.py"))
    a = ap.parse_args()

    if not os.path.exists(a.canonical):
        print(f"VERIFY BASELINE: FAIL -- canonical script not found: {a.canonical}")
        return 1

    ours = top_level_defs(a.ours)
    canon = top_level_defs(a.canonical)

    problems = []
    for name in FROZEN:
        if name not in ours:
            problems.append(f"{name}: missing from {a.ours}")
            continue
        if name not in canon:
            problems.append(f"{name}: missing from {a.canonical}")
            continue
        if ast.dump(ours[name]) != ast.dump(canon[name]):
            problems.append(f"{name}: AST differs from canonical")

    # Budget defaults -- the thing that actually changed on 2026-08-27.
    if "parse_args" in ours:
        d = arg_defaults(ours["parse_args"])
        for k, v in EXPECT_DEFAULTS.items():
            if d.get(k) != v:
                problems.append(
                    f"parse_args --{k} default is {d.get(k)!r}, expected {v!r}")
        cd = arg_defaults(canon["parse_args"]) if "parse_args" in canon else {}
        for k in EXPECT_DEFAULTS:
            if k in cd and k in d and cd[k] != d[k]:
                problems.append(
                    f"parse_args --{k}: ours {d[k]!r} != canonical {cd[k]!r}")

    if problems:
        print("VERIFY BASELINE: FAIL")
        for p in problems:
            print(f"  - {p}")
        print(f"\n  ours:      {a.ours}")
        print(f"  canonical: {a.canonical}")
        print("  If the judges changed the harness, port the change into "
              "benchmark.py and re-run tools/sync_entrypoint.py.")
        return 1

    print(f"VERIFY BASELINE: pass -- {len(FROZEN)} frozen symbols match "
          f"{os.path.basename(a.canonical)}; atol=0.002 rtol=0.02 confirmed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
