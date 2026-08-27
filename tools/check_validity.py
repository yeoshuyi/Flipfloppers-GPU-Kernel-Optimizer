#!/usr/bin/env python3
"""
Static validity gate. Replaces the 'adversary' agent for the ~90% of
benchmark-gaming that is mechanically detectable. Costs zero tokens.

Run BEFORE run_accuracy. Exit 1 blocks the candidate.

    python3 tools/check_validity.py src/model.py [--speedup 12.4] [--shape B8_S128]

What this cannot catch (the residual ~10%) is listed at the bottom and needs
your eyes, once, at solidification -- not an LLM on every iteration.
"""
import ast
import re
import sys
import argparse

# FP8 theoretical floor at default shape, ms. Nothing can legitimately beat it.
FP8_FLOOR_MS = {"B8_S128": 0.122}
BASELINE_REF_MS = {"B8_S128": 2.0}   # calibrate from your Phase-0 sweep

FORBIDDEN_CALLS = {
    "detach_": "in-place detach on the output path",
}


def check_source(path: str) -> list[str]:
    src = open(path).read()
    tree = ast.parse(src)
    bad: list[str] = []

    # --- data_ptr used outside the mask-cache whitelist -------------------
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "data_ptr"):
            continue
        fn = _enclosing_func(tree, node)
        if fn != "_mask_is_all_ones":
            bad.append(
                f"data_ptr() called in {fn or '<module>'}; only "
                f"_mask_is_all_ones may use it (caches a mask property, "
                f"never output)")

    # --- new Parameters/Buffers break load_state_dict(strict=True) --------
    for m in re.finditer(r"register_buffer\s*\(|nn\.Parameter\s*\(", src):
        line = src[:m.start()].count("\n") + 1
        bad.append(f"line {line}: Parameter/Buffer registration -- "
                   f"strict=True will reject the new key. Use a plain attribute.")

    # --- explicit attn_mask kicks SDPA off the flash backend --------------
    # A masked call is only a problem if it is the ONLY way the file ever
    # calls SDPA -- i.e. there is no coexisting is_causal-only fast path for
    # the unpadded case. A masked call that sits alongside a real fast path
    # is the PADDED-regime modifier (CLAUDE.md dispatch table), not a lazy
    # attn_mask default that skips is_causal.
    sdpa_calls = re.findall(r"scaled_dot_product_attention\s*\((.*?)\)",
                             src, re.S)
    def _is_masked(call: str) -> bool:
        return bool(re.search(r"attn_mask\s*=\s*(?!None\b)", call))
    def _is_fast_path(call: str) -> bool:
        return "is_causal" in call and not _is_masked(call)
    masked_calls = [c for c in sdpa_calls if _is_masked(c)]
    if masked_calls and not any(_is_fast_path(c) for c in sdpa_calls):
        bad.append("explicit attn_mask passed to SDPA with no coexisting "
                   "is_causal fast path -- forces the slow math backend on "
                   "every call. Use is_causal=True for the unpadded case.")

    # --- module-level mutable caches that could hold outputs --------------
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, (ast.Dict, ast.List))
                and _is_module_level(tree, node)):
            for t in node.targets:
                name = getattr(t, "id", "")
                if name and "mask" not in name.lower():
                    bad.append(f"module-level mutable '{name}' -- verify it "
                               f"cannot cache outputs across calls")

    for call, why in FORBIDDEN_CALLS.items():
        if re.search(rf"\.{call}\s*\(", src):
            bad.append(f"{call}(): {why}")

    return bad


def check_result(speedup: float | None, shape: str | None) -> list[str]:
    bad = []
    if speedup and shape and shape in FP8_FLOOR_MS:
        implied = BASELINE_REF_MS[shape] / speedup
        if implied < FP8_FLOOR_MS[shape]:
            bad.append(
                f"implied latency {implied:.4f} ms beats the FP8 theoretical "
                f"floor {FP8_FLOOR_MS[shape]:.3f} ms at {shape}. "
                f"The candidate is not computing the answer.")
    return bad


def _enclosing_func(tree, target):
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(fn):
                if sub is target:
                    return fn.name
    return None


def _is_module_level(tree, node):
    return any(node is child for child in tree.body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--speedup", type=float)
    ap.add_argument("--shape")
    a = ap.parse_args()

    issues = check_source(a.path) + check_result(a.speedup, a.shape)

    if issues:
        print("VALIDITY GATE: FAIL")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)

    print("VALIDITY GATE: pass")
    if a.path.endswith("benchmark.py"):
        print("  note: also run tools/verify_baseline.py before a ship "
              "(frozen harness == judges' torch_transformer_benchmark.py)")
    sys.exit(0)

# ---------------------------------------------------------------------------
# NOT machine-checkable -- review by hand ONCE, at solidification:
#   * preconditions asserted but not runtime-checked
#   * a fast path whose fallback is also wrong
#   * dispatch thresholds that happen to match only the disclosed shapes
#   * numerical shortcuts valid for randn inputs but not in general
# These are cheap to eyeball on a final diff and expensive to re-litigate
# with an LLM on every iteration.
# ---------------------------------------------------------------------------
