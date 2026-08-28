#!/usr/bin/env python3
"""Generate the README figures as static SVGs (stdlib only -- the container has
no matplotlib). Numbers are frozen (optimization loop converged at PROGRESS
step 52); each block cites the results log it came from.

    python3 tools/make_figures.py        # writes assets/*.svg

Charts:
  assets/latency_breakdown.svg  per-shape stage split (SDPA / GEMM / GELU / LN+res)
  assets/pareto_accuracy.svg    whole-model speedup vs accuracy-budget usage,
                                with the rejected precision tiers past the wall
  assets/roofline.svg           shipped vs accuracy-legal roofline, 4 shapes
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets")

# ---- palette (readable on GitHub light and dark; light card either way) ----
BG = "#ffffff"
INK = "#1b1f24"
SUB = "#5b6570"
GRID = "#d9dee4"
C_SDPA, C_GEMM, C_GELU, C_LN = "#3b6ea5", "#6a8d4e", "#b0762f", "#9aa0a8"
C_WALL = "#b5443a"
C_OK = "#3c7a56"
C_ROOF = "#9aa0a8"
FONT = ("font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,"
        "Helvetica,Arial,sans-serif")


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def T(x, y, s, size=13, fill=INK, anchor="start", weight="400", mono=False):
    fam = "font-family:ui-monospace,SFMono-Regular,Menlo,monospace" if mono else FONT
    return (f'<text x="{x:.1f}" y="{y:.1f}" style="{fam};font-size:{size}px;'
            f'font-weight:{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')


def R(x, y, w, h, fill, rx=0, opacity=1.0):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" '
            f'height="{max(h,0):.1f}" rx="{rx}" fill="{fill}" opacity="{opacity}"/>')


def LN_(x1, y1, x2, y2, stroke=GRID, w=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w}"{d}/>')


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{esc(title)}">'
            f'{R(0,0,w,h,BG)}{body}</svg>\n')


# ===========================================================================
# 1. latency breakdown  --  docs/FINAL_SCORECARD.md 2 / results/logs/final_scorecard_run171.log
# ===========================================================================
# row, shipped_ms, SDPA%, GEMM%, GELU%, LN+res%, bound-by
LAT = [
    (1,  0.211, 14.1, 57.4, 5.9, 22.5, "small-GEMM fill + launch"),
    (2,  0.078, 23.0, 50.5, 5.7, 20.8, "launch (kernel-body floor)"),
    (3,  0.088, 20.9, 49.5, 5.3, 24.3, "launch"),
    (4,  0.112, 17.2, 52.0, 5.7, 25.1, "launch"),
    (5,  0.373, 14.3, 53.2, 5.6, 26.8, "small-GEMM fill"),
    (6,  52.49, 20.3, 32.6, 8.2, 38.9, "memory bandwidth"),
    (7,  0.103, 30.6, 40.1, 6.0, 23.3, "SDPA + launch (d=32)"),
    (8,  4.327, 6.5,  72.5, 2.9, 18.0, "compute -- 165 TFLOP/s roofline"),
    (9,  0.210, 13.4, 57.9, 6.0, 22.8, "small-GEMM fill + launch"),
    (10, 0.213, 14.7, 56.9, 5.9, 22.5, "small-GEMM fill + launch"),
    (11, 0.286, 36.5, 42.4, 4.4, 16.7, "attention (16 heads)"),
    (12, 0.123, 24.9, 47.3, 5.1, 22.7, "launch (S=32)"),
    (13, 2.209, 32.6, 26.3, 6.1, 35.0, "SDPA O(S^2) + memory"),
]


def fig_latency():
    W, H = 900, 560
    x0, x1 = 200, 640          # bar track
    top, rowh = 96, 30
    b = []
    b.append(T(36, 40, "Where the shipped forward spends its time", 18, INK, weight="600"))
    b.append(T(36, 62, "per official causal shape, % of the graphed 4-layer forward "
                        "(sums to shipped ms at right)", 12, SUB))
    # legend
    lx = 200
    for lab, col in [("SDPA", C_SDPA), ("GEMM (QKV+proj+FFN)", C_GEMM),
                     ("GELU", C_GELU), ("LayerNorm + residual + cast", C_LN)]:
        b.append(R(lx, 74, 11, 11, col, rx=2))
        b.append(T(lx + 16, 84, lab, 11, SUB))
        lx += 20 + len(lab) * 6.6 + 16
    for i, (row, ms, s, g, ge, ln, bound) in enumerate(LAT):
        y = top + i * rowh
        b.append(T(x0 - 12, y + 15, f"row {row}", 12, INK, anchor="end", mono=True))
        cx = x0
        for frac, col in ((s, C_SDPA), (g, C_GEMM), (ge, C_GELU), (ln, C_LN)):
            w = (x1 - x0) * frac / 100.0
            b.append(R(cx, y + 3, w, 18, col))
            cx += w
        b.append(T(x1 + 12, y + 16, f"{ms:>8.3f} ms", 12, INK, mono=True))
        b.append(T(x1 + 92, y + 16, bound, 11, SUB))
    b.append(LN_(x0, top - 6, x0, top + len(LAT) * rowh - 6, GRID, 1))
    b.append(T(36, H - 16, "source: results/logs/final_scorecard_run171.log  "
                           "(job 171, per-row clean torch._dynamo state)", 10, SUB, mono=True))
    return svg(W, H, "".join(b), "latency breakdown per shape")


# ===========================================================================
# 2. pareto: speed vs accuracy-budget usage
#    shipped: docs/FINAL_SCORECARD.md 1 ; rejected tiers: docs/DOCUMENTATION.md 4
# ===========================================================================
# row, max_abs, speedup
SHIP = [
    (1, 0.00137, 4.96), (2, 0.00137, 13.64), (3, 0.00137, 11.84), (4, 0.00137, 9.38),
    (5, 0.00137, 4.57), (6, 0.00195, 5.54), (7, 0.00211, 9.87), (8, 0.00141, 1.93),
    (9, 0.00145, 4.62), (10, 0.00138, 4.92), (11, 0.00137, 15.35), (12, 0.00141, 8.44),
    (13, 0.00137, 31.76),
]
BUDGET = 0.002
# rejected tier, xmult (max_abs / 0.002), label
REJECT = [
    (1.95, "FP16-accumulate GEMM, K=128"),
    (5.5,  "BF16 whole model"),
    (10.5, "FP16-accumulate GEMM, K=1024"),
    (15.0, "INT8 FFN"),
    (33.0, "FP8 FFN"),
]
import math


def fig_pareto():
    W, H = 900, 520
    L, Rr, Tp, Bt = 70, 40, 70, 70
    pw, ph = W - L - Rr, H - Tp - Bt
    # x: log10 of (% of budget), from 40% to 6000%
    xmin, xmax = math.log10(40), math.log10(6000)
    ymin, ymax = 0, 34

    def px(pct):
        return L + pw * (math.log10(pct) - xmin) / (xmax - xmin)

    def py(v):
        return Tp + ph * (1 - (v - ymin) / (ymax - ymin))

    b = []
    b.append(T(36, 38, "Anything faster is past the accuracy wall", 18, INK, weight="600"))
    b.append(T(36, 58, "whole-model speedup vs. how much of the atol=0.002 budget "
                       "the stack's worst element uses", 12, SUB))
    # forbidden band
    b.append(R(px(100), Tp, (L + pw) - px(100), ph, C_WALL, opacity=0.06))
    b.append(LN_(px(100), Tp, px(100), Tp + ph, C_WALL, 2, dash="5 4"))
    b.append(T(px(100) + 6, Tp + 14, "atol budget (100%)", 11, C_WALL, weight="600"))
    b.append(T(px(100) + 6, Tp + 30, "gate is failed==0, not max_abs", 10, SUB))
    # x gridlines
    for pct in (50, 100, 200, 500, 1000, 3000):
        gx = px(pct)
        b.append(LN_(gx, Tp, gx, Tp + ph, GRID, 1))
        b.append(T(gx, Tp + ph + 18, f"{pct}%", 11, SUB, anchor="middle", mono=True))
    for v in (0, 10, 20, 30):
        gy = py(v)
        b.append(LN_(L, gy, L + pw, gy, GRID, 1))
        b.append(T(L - 10, gy + 4, f"{v}x", 11, SUB, anchor="end", mono=True))
    b.append(T(L + pw / 2, H - 22, "worst-element error, % of the 0.002 budget (log scale)",
              12, SUB, anchor="middle"))
    b.append(T(20, Tp + ph / 2, "speedup", 12, SUB, anchor="middle"))
    # shipped points
    for row, ma, sp in SHIP:
        cx, cy = px(ma / BUDGET * 100), py(sp)
        b.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{C_OK}"/>')
        b.append(T(cx + 8, cy + 4, f"{row}", 10, C_OK, mono=True))
    b.append(T(px(70), py(33), "13/13 shipped shapes  (pass: failed==0)", 12, C_OK, weight="600"))
    # rejected tiers
    for xm, lab in REJECT:
        cx = px(xm * 100)
        b.append(f'<path d="M {cx-5:.1f} {py(2)-5:.1f} l 10 0 l -5 9 z" fill="{C_WALL}"/>')
        b.append(T(cx, py(2) - 12, lab, 10, C_WALL, anchor="middle"))
    b.append(T(36, H - 14, "source: results/logs/official_causal_sweep_run168.log ; "
                           "rejected tiers: docs/DOCUMENTATION.md 4 (BF16/FP8/INT8 20-seed probes)",
              10, SUB, mono=True))
    return svg(W, H, "".join(b), "speed vs accuracy pareto")


# ===========================================================================
# 3. roofline: shipped vs accuracy-legal floor  --  docs/FINAL_SCORECARD.md 3
# ===========================================================================
# row, shipped_ms, roofline_ms, ratio, binding wall
ROOF = [
    (1,  0.211, 0.069, "3.1x", "launch + sub-roofline small GEMM"),
    (13, 2.209, 1.030, "2.1x", "SDPA O(S^2) + LayerNorm/residual traffic"),
    (6,  52.49, 25.70, "2.0x", "memory bandwidth (elementwise at the BW roofline)"),
    (8,  4.327, 2.780, "1.6x", "compute -- cuBLAS at 94-96% of the 165 TFLOP/s roofline"),
]


def fig_roofline():
    W, H = 900, 380
    x0, x1 = 150, 560
    top, rowh = 100, 56
    b = []
    b.append(T(36, 40, "How close to the practical hardware ceiling", 18, INK, weight="600"))
    b.append(T(36, 62, "shipped latency vs. the accuracy-legal roofline "
                       "(GEMM FLOPs / 165.2 TFLOP/s + measured SDPA + kernel-body floor)",
              12, SUB))
    b.append(R(150, 74, 11, 11, C_OK, rx=2)); b.append(T(166, 84, "shipped", 11, SUB))
    b.append(R(232, 74, 11, 11, C_ROOF, rx=2)); b.append(T(248, 84, "roofline (unreachable floor)", 11, SUB))
    mx = max(r[1] for r in ROOF)
    for i, (row, ms, rf, ratio, wall) in enumerate(ROOF):
        y = top + i * rowh
        b.append(T(x0 - 12, y + 12, f"row {row}", 12, INK, anchor="end", mono=True))
        wship = (x1 - x0) * ms / mx
        wroof = (x1 - x0) * rf / mx
        b.append(R(x0, y, wship, 14, C_OK))
        b.append(R(x0, y + 17, wroof, 10, C_ROOF))
        b.append(T(x0 + max(wship, wroof) + 10, y + 13,
                   f"{ms:.3f} ms  =  {ratio} of roofline", 12, INK, mono=True))
        b.append(T(x0, y + 40, wall, 11, SUB))
    b.append(T(36, H - 16, "source: docs/FINAL_SCORECARD.md 3 ; docs/PARETO_FRONTIER_ANALYSIS.md "
                           "-- \"the stack is at the frontier\"", 10, SUB, mono=True))
    return svg(W, H, "".join(b), "roofline proximity")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("latency_breakdown", fig_latency),
                     ("pareto_accuracy", fig_pareto),
                     ("roofline", fig_roofline)]:
        p = os.path.join(OUT, name + ".svg")
        with open(p, "w") as f:
            f.write(fn())
        print("wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
