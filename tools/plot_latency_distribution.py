#!/usr/bin/env python3
"""Turn one or more tools/latency_distribution_sweep.py JSON logs into a single
self-contained HTML chart: per-shape latency distributions, normalized to each
series' own mean/SD (z-score) so shapes with wildly different absolute latency
(a few ms vs. hundreds of ms) can be compared on one x-axis, aligned at 0.

Stdlib only (matches tools/make_figures.py's convention -- no charting library
in the container/venv). Row 14 and rows 1-13 are typically two separate JSON
files (see tools/latency_distribution_sweep.py's docstring for why); pass both:

    python3 tools/plot_latency_distribution.py \\
        results/artifacts/latency_distribution_<stamp>_rows1-13.json \\
        results/artifacts/latency_distribution_<stamp>_row14.json \\
        --out results/artifacts/latency_distribution.html
"""
from __future__ import annotations

import argparse
import colorsys
import json
import statistics
from typing import Dict, List, Tuple

N_SHAPES = 14
BIN_WIDTH = 0.25
Z_RANGE = 4.0  # bins span [-Z_RANGE, +Z_RANGE], samples beyond are clipped in


def make_palette(n: int) -> List[Dict[str, str]]:
    """n evenly-spaced hues around the wheel -- the maximum-min-spacing
    arrangement for a fixed, known series count (see dataviz skill:
    color-formula.md). Saturation/lightness bands follow the skill's
    documented light/dark categorical bands."""
    palette = []
    for i in range(n):
        hue = (i * 360.0 / n) % 360.0
        r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.46, 0.62)
        light = "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))
        r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.60, 0.62)
        dark = "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))
        palette.append({"light": light, "dark": dark})
    return palette


def bin_edges() -> List[float]:
    n_bins = int(round(2 * Z_RANGE / BIN_WIDTH))
    return [-Z_RANGE + i * BIN_WIDTH for i in range(n_bins + 1)]


def histogram(samples_ms: List[float], mean_ms: float, sd_ms: float) -> List[Tuple[float, float]]:
    edges = bin_edges()
    n_bins = len(edges) - 1
    counts = [0] * n_bins
    n = len(samples_ms)
    for v in samples_ms:
        z = 0.0 if sd_ms == 0 else (v - mean_ms) / sd_ms
        z = max(-Z_RANGE, min(Z_RANGE, z))  # clip fat-tail outliers into the edge bins
        idx = min(int((z + Z_RANGE) / BIN_WIDTH), n_bins - 1)
        counts[idx] += 1
    return [(edges[i] + BIN_WIDTH / 2, counts[i] / n if n else 0.0) for i in range(n_bins)]


def standard_normal_reference() -> List[Tuple[float, float]]:
    """Expected relative-frequency-per-bin for a standard normal, so a
    series' departure from Gaussian is visible against a reference curve."""
    import math

    edges = bin_edges()
    n_bins = len(edges) - 1
    out = []
    for i in range(n_bins):
        center = edges[i] + BIN_WIDTH / 2
        pdf = math.exp(-0.5 * center * center) / math.sqrt(2 * math.pi)
        out.append((center, pdf * BIN_WIDTH))
    return out


def shape_label(shape: dict) -> str:
    return (
        f"B={shape['batch_size']} d={shape['d_model']} H={shape['heads']} "
        f"S={shape['seq_len']} L={shape['layers']} F={shape['ffn_dim']}"
    )


def load_series(paths: List[str]) -> List[dict]:
    series = []
    for path in paths:
        with open(path) as f:
            payload = json.load(f)
        series.extend(payload["series"])
    return series


def build_series_payload(series: List[dict]) -> List[dict]:
    palette = make_palette(N_SHAPES)
    out = []
    for rec in series:
        samples = rec["samples_ms"]
        mean_ms = rec.get("mean_ms") or statistics.fmean(samples)
        sd_ms = rec.get("sd_ms")
        if sd_ms is None:
            sd_ms = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        bins = histogram(samples, mean_ms, sd_ms)
        color = palette[(rec["row"] - 1) % N_SHAPES]
        cv_pct = (sd_ms / mean_ms * 100.0) if mean_ms else 0.0
        out.append(
            {
                "row": rec["row"],
                "variant": rec["variant"],
                "shape": shape_label(rec["shape"]),
                "n": rec["n"],
                "mean_ms": mean_ms,
                "sd_ms": sd_ms,
                "cv_pct": cv_pct,
                "color": color,
                "bins": [[round(x, 4), round(y, 6)] for x, y in bins],
            }
        )
    out.sort(key=lambda r: (r["row"], r["variant"]))
    return out


PAGE_TEMPLATE = """<title>Latency Spectrograph</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --surface-1: #f7f8fa;
  --page-bg: #eef1f4;
  --text-primary: #10151c;
  --text-secondary: #4a5568;
  --text-muted: #7c8798;
  --gridline: #dde3ea;
  --axis: #b9c2cd;
  --border: rgba(16,21,28,0.09);
  --accent: #b8671b;
  --accent-strong: #8f4f14;
  --seq-blue-300: #6da7ec;
  --seq-blue-600: #184f95;
  --font-sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --surface-1: #12161c;
    --page-bg: #0a0d11;
    --text-primary: #f3f5f7;
    --text-secondary: #b9c2cd;
    --text-muted: #7c8798;
    --gridline: #232a33;
    --axis: #384352;
    --border: rgba(255,255,255,0.08);
    --accent: #e2a24d;
    --accent-strong: #f0bb74;
    --seq-blue-300: #3987e5;
    --seq-blue-600: #86b6ef;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface-1: #12161c;
  --page-bg: #0a0d11;
  --text-primary: #f3f5f7;
  --text-secondary: #b9c2cd;
  --text-muted: #7c8798;
  --gridline: #232a33;
  --axis: #384352;
  --border: rgba(255,255,255,0.08);
  --accent: #e2a24d;
  --accent-strong: #f0bb74;
  --seq-blue-300: #3987e5;
  --seq-blue-600: #86b6ef;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--page-bg);
  color: var(--text-primary);
  font: 400 14px/1.5 var(--font-sans);
  margin: 0;
  padding: 24px;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; }}
h1 {{
  font-size: 21px;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0 0 6px;
  text-wrap: balance;
  border-left: 3px solid var(--accent);
  padding-left: 10px;
}}
.subtitle {{ color: var(--text-secondary); margin: 0 0 4px 13px; font-size: 13px; }}
.note {{ color: var(--text-muted); font-size: 12px; margin: 0 0 20px 13px; max-width: 80ch; }}
.card {{
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 20px;
}}
.controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px 16px;
  align-items: center;
  margin-bottom: 12px;
  font-size: 12px;
  color: var(--text-secondary);
}}
.controls label {{ display: inline-flex; align-items: center; gap: 5px; cursor: pointer; }}
.controls input[type="checkbox"] {{ accent-color: var(--accent); }}
.controls button {{
  font: inherit;
  font-size: 12px;
  color: var(--text-primary);
  background: var(--page-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 4px 10px;
  cursor: pointer;
}}
.controls button:hover {{ border-color: var(--accent); color: var(--accent-strong); }}
.chart-area {{ position: relative; }}
svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
.gridline {{ stroke: var(--gridline); stroke-width: 1; }}
.axis-line {{ stroke: var(--axis); stroke-width: 1; }}
.axis-label {{ fill: var(--text-muted); font-size: 11px; font-family: var(--font-mono); }}
.axis-title {{ fill: var(--text-secondary); font-size: 12px; }}
.series-path {{ fill: none; stroke-width: 2; transition: opacity 120ms; }}
.ref-path {{ fill: none; stroke: var(--text-muted); stroke-width: 1.5; stroke-dasharray: 2,3; }}
.crosshair {{ stroke: var(--axis); stroke-width: 1; pointer-events: none; }}
.legend {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 4px 10px;
  margin-top: 14px;
  font-size: 12px;
}}
.legend-item {{
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  padding: 3px 4px;
  border-radius: 5px;
  color: var(--text-secondary);
  user-select: none;
}}
.legend-item:hover {{ background: var(--page-bg); color: var(--text-primary); }}
.legend-item.off {{ opacity: 0.35; }}
.legend-key {{ width: 20px; height: 2px; flex: none; border-radius: 1px; }}
.legend-key.dashed {{ background: none; border-top: 2px dashed currentColor; height: 0; }}
.tooltip {{
  position: absolute;
  pointer-events: none;
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 10px;
  font-size: 11.5px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.18);
  min-width: 160px;
  max-width: 260px;
  visibility: hidden;
  z-index: 5;
}}
.tooltip .t-x {{ color: var(--text-muted); margin-bottom: 5px; }}
.tooltip-row {{ display: flex; align-items: center; gap: 6px; padding: 1px 0; }}
.tooltip-row .k {{ width: 14px; height: 2px; flex: none; border-radius: 1px; }}
.tooltip-row .lbl {{ color: var(--text-secondary); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.tooltip-row .val {{ color: var(--text-primary); font-family: var(--font-mono); font-weight: 500; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--gridline); }}
th {{ color: var(--text-muted); font-weight: 600; cursor: pointer; white-space: nowrap; text-transform: uppercase; letter-spacing: 0.03em; font-size: 11px; }}
th:hover {{ color: var(--accent-strong); }}
td.num, th.num {{ text-align: right; }}
td.num {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}
.cv-bar-wrap {{ display: flex; align-items: center; gap: 8px; justify-content: flex-end; }}
.cv-bar {{ height: 8px; border-radius: 2px; background: var(--seq-blue-300); }}
.swatch {{ width: 10px; height: 10px; border-radius: 50%; flex: none; }}
</style>
<div class="wrap">
  <h1>Latency spectrograph</h1>
  <p class="subtitle">
    Per-shape latency distributions across the official causal matrix &mdash;
    {n_series} series over {n_rows} shapes (baseline vs. optimized). Each
    series is z-scored against its own mean/SD, so every curve is aligned at
    0 regardless of its absolute latency scale.
  </p>
  <p class="note">
    X-axis: (sample &minus; mean) / SD for that (shape, variant). Y-axis:
    fraction of that series' samples falling in each {bin_width}&sigma;-wide
    bin. The dashed gray curve is the expected shape for a normal distribution
    with the same binning, as a reference for spotting skew or heavy tails.
    {row14_note}
  </p>

  <div class="card">
    <div class="controls">
      <span>Show:</span>
      <label><input type="checkbox" id="toggle-optimized" checked> Optimized</label>
      <label><input type="checkbox" id="toggle-baseline" checked> Baseline</label>
      <label><input type="checkbox" id="toggle-ref" checked> Normal reference</label>
      <button id="btn-all">All shapes</button>
      <button id="btn-none">No shapes</button>
    </div>
    <div class="chart-area">
      <svg id="chart" viewBox="0 0 920 460" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="tooltip" id="tooltip"></div>
    </div>
    <div class="legend" id="legend"></div>
  </div>

  <div class="card">
    <table id="data-table">
      <thead>
        <tr>
          <th data-key="row" class="num">Row</th>
          <th data-key="variant">Variant</th>
          <th data-key="shape">Shape</th>
          <th data-key="n" class="num">n</th>
          <th data-key="mean_ms" class="num">Mean (ms)</th>
          <th data-key="sd_ms" class="num">SD (ms)</th>
          <th data-key="cv_pct" class="num">CV (SD/mean)</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
</div>
<script>
const SERIES = {series_json};
const REF = {ref_json};
const Z_RANGE = {z_range};
const BIN_WIDTH = {bin_width};

const state = {{
  rows: new Set(SERIES.map(s => s.row)),
  showOptimized: true,
  showBaseline: true,
  showRef: true,
}};

function isVisible(s) {{
  if (!state.rows.has(s.row)) return false;
  if (s.variant === "baseline" && !state.showBaseline) return false;
  if (s.variant === "optimized" && !state.showOptimized) return false;
  return true;
}}

// ---- chart geometry -------------------------------------------------------
const svg = document.getElementById("chart");
const SVG_NS = "http://www.w3.org/2000/svg";
const PAD = {{ l: 46, r: 16, t: 14, b: 40 }};
const W = 920, H = 460;
const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b;

let maxFreq = 0.02;
for (const s of SERIES) for (const [, y] of s.bins) if (y > maxFreq) maxFreq = y;
for (const [, y] of REF) if (y > maxFreq) maxFreq = y;
maxFreq *= 1.12;

function isDarkMode() {{
  const attr = document.documentElement.getAttribute("data-theme");
  if (attr === "dark") return true;
  if (attr === "light") return false;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
}}
function seriesColor(s) {{ return isDarkMode() ? s.color.dark : s.color.light; }}

function xPix(z) {{ return PAD.l + (z + Z_RANGE) / (2 * Z_RANGE) * plotW; }}
function yPix(f) {{ return PAD.t + plotH - (f / maxFreq) * plotH; }}

function pathFor(bins) {{
  return bins.map(([x, y], i) => (i === 0 ? "M" : "L") + xPix(x).toFixed(1) + "," + yPix(y).toFixed(1)).join(" ");
}}

function el(tag, attrs) {{
  const e = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}}

function render() {{
  svg.textContent = "";

  // gridlines + axis ticks
  for (let z = -Z_RANGE; z <= Z_RANGE; z += 1) {{
    const x = xPix(z);
    svg.appendChild(el("line", {{ x1: x, x2: x, y1: PAD.t, y2: PAD.t + plotH, class: "gridline" }}));
    const t = el("text", {{ x: x, y: PAD.t + plotH + 16, class: "axis-label", "text-anchor": "middle" }});
    t.textContent = (z > 0 ? "+" : "") + z + "σ";
    svg.appendChild(t);
  }}
  const axisTitleX = el("text", {{ x: PAD.l + plotW / 2, y: H - 4, class: "axis-title", "text-anchor": "middle" }});
  axisTitleX.textContent = "standard deviations from that series' own mean";
  svg.appendChild(axisTitleX);

  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {{
    const f = (maxFreq * i) / yTicks;
    const y = yPix(f);
    svg.appendChild(el("line", {{ x1: PAD.l, x2: PAD.l + plotW, y1: y, y2: y, class: "gridline" }}));
    const t = el("text", {{ x: PAD.l - 8, y: y + 4, class: "axis-label", "text-anchor": "end" }});
    t.textContent = (f * 100).toFixed(1) + "%";
    svg.appendChild(t);
  }}
  const axisTitleY = el("text", {{
    x: -(PAD.t + plotH / 2), y: 14, class: "axis-title", "text-anchor": "middle",
    transform: "rotate(-90)",
  }});
  axisTitleY.textContent = "share of samples per bin";
  svg.appendChild(axisTitleY);

  svg.appendChild(el("line", {{ x1: PAD.l, x2: PAD.l + plotW, y1: PAD.t + plotH, y2: PAD.t + plotH, class: "axis-line" }}));
  svg.appendChild(el("line", {{ x1: PAD.l, x2: PAD.l, y1: PAD.t, y2: PAD.t + plotH, class: "axis-line" }}));

  if (state.showRef) {{
    svg.appendChild(el("path", {{ d: pathFor(REF), class: "ref-path" }}));
  }}

  for (const s of SERIES) {{
    const visible = isVisible(s);
    const path = el("path", {{
      d: pathFor(s.bins),
      class: "series-path",
      stroke: seriesColor(s),
    }});
    path.style.opacity = visible ? "0.85" : "0";
    if (s.variant === "baseline") path.setAttribute("stroke-dasharray", "6,4");
    path.dataset.row = s.row;
    path.dataset.variant = s.variant;
    svg.appendChild(path);
  }}

  renderLegend();
  renderTable();
}}

// ---- legend -----------------------------------------------------------
function renderLegend() {{
  const legend = document.getElementById("legend");
  legend.textContent = "";
  const rows = [...new Set(SERIES.map(s => s.row))].sort((a, b) => a - b);
  for (const row of rows) {{
    const rep = SERIES.find(s => s.row === row);
    const item = document.createElement("div");
    item.className = "legend-item" + (state.rows.has(row) ? "" : " off");
    const key = document.createElement("span");
    key.className = "legend-key";
    key.style.background = seriesColor(rep);
    const label = document.createElement("span");
    label.textContent = "Row " + row + " (" + rep.shape + ")";
    item.appendChild(key);
    item.appendChild(label);
    item.addEventListener("click", () => {{
      if (state.rows.has(row)) state.rows.delete(row); else state.rows.add(row);
      render();
    }});
    legend.appendChild(item);
  }}
}}

// ---- data table ---------------------------------------------------------
let sortKey = "row", sortAsc = true;
function renderTable() {{
  const tbody = document.querySelector("#data-table tbody");
  tbody.textContent = "";
  const rows = [...SERIES].sort((a, b) => {{
    const av = a[sortKey], bv = b[sortKey];
    const cmp = typeof av === "string" ? av.localeCompare(bv) : av - bv;
    return sortAsc ? cmp : -cmp;
  }});
  const maxCv = Math.max(...SERIES.map(s => s.cv_pct), 0.001);
  for (const s of rows) {{
    const tr = document.createElement("tr");

    const tdRow = document.createElement("td"); tdRow.className = "num"; tdRow.textContent = s.row;
    const tdVariant = document.createElement("td"); tdVariant.textContent = s.variant;
    const tdShape = document.createElement("td");
    const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = seriesColor(s);
    sw.style.display = "inline-block"; sw.style.marginRight = "6px";
    tdShape.appendChild(sw);
    tdShape.appendChild(document.createTextNode(s.shape));
    const tdN = document.createElement("td"); tdN.className = "num"; tdN.textContent = s.n;
    const tdMean = document.createElement("td"); tdMean.className = "num"; tdMean.textContent = s.mean_ms.toFixed(3);
    const tdSd = document.createElement("td"); tdSd.className = "num"; tdSd.textContent = s.sd_ms.toFixed(4);
    const tdCv = document.createElement("td"); tdCv.className = "num";
    const cvWrap = document.createElement("div"); cvWrap.className = "cv-bar-wrap";
    const bar = document.createElement("div"); bar.className = "cv-bar";
    bar.style.width = Math.max(2, (s.cv_pct / maxCv) * 60) + "px";
    const cvText = document.createElement("span"); cvText.textContent = s.cv_pct.toFixed(2) + "%";
    cvWrap.appendChild(bar); cvWrap.appendChild(cvText);
    tdCv.appendChild(cvWrap);

    tr.append(tdRow, tdVariant, tdShape, tdN, tdMean, tdSd, tdCv);
    tbody.appendChild(tr);
  }}
}}

document.querySelectorAll("#data-table th").forEach(th => {{
  th.addEventListener("click", () => {{
    const key = th.dataset.key;
    if (sortKey === key) sortAsc = !sortAsc; else {{ sortKey = key; sortAsc = true; }}
    renderTable();
  }});
}});

// ---- hover crosshair + tooltip -------------------------------------------
const tooltip = document.getElementById("tooltip");
let crosshairEl = null;

function valueAt(bins, z) {{
  let best = bins[0];
  for (const b of bins) if (Math.abs(b[0] - z) < Math.abs(best[0] - z)) best = b;
  return best[1];
}}

svg.addEventListener("pointermove", (evt) => {{
  const rect = svg.getBoundingClientRect();
  const scaleX = W / rect.width;
  const px = (evt.clientX - rect.left) * scaleX;
  if (px < PAD.l || px > PAD.l + plotW) {{ tooltip.style.visibility = "hidden"; if (crosshairEl) crosshairEl.remove(); return; }}
  const z = ((px - PAD.l) / plotW) * (2 * Z_RANGE) - Z_RANGE;

  if (crosshairEl) crosshairEl.remove();
  crosshairEl = el("line", {{ x1: px, x2: px, y1: PAD.t, y2: PAD.t + plotH, class: "crosshair" }});
  svg.appendChild(crosshairEl);

  const rows = SERIES.filter(isVisible)
    .map(s => ({{ s, y: valueAt(s.bins, z) }}))
    .sort((a, b) => b.y - a.y)
    .slice(0, 10);

  tooltip.textContent = "";
  const head = document.createElement("div");
  head.className = "t-x";
  head.textContent = (z >= 0 ? "+" : "") + z.toFixed(2) + "σ";
  tooltip.appendChild(head);
  for (const {{ s, y }} of rows) {{
    const r = document.createElement("div");
    r.className = "tooltip-row";
    const k = document.createElement("span");
    k.className = "k";
    const c = seriesColor(s);
    k.style.background = c;
    if (s.variant === "baseline") {{ k.style.background = "none"; k.style.borderTop = "2px dashed " + c; }}
    const lbl = document.createElement("span");
    lbl.className = "lbl";
    lbl.textContent = "Row " + s.row + " " + s.variant;
    const val = document.createElement("span");
    val.className = "val";
    val.textContent = (y * 100).toFixed(1) + "%";
    r.append(k, lbl, val);
    tooltip.appendChild(r);
  }}

  const rectArea = svg.parentElement.getBoundingClientRect();
  const leftPx = (px / scaleX);
  tooltip.style.left = Math.min(leftPx + 14, rectArea.width - 220) + "px";
  tooltip.style.top = "8px";
  tooltip.style.visibility = rows.length ? "visible" : "hidden";
}});
svg.addEventListener("pointerleave", () => {{
  tooltip.style.visibility = "hidden";
  if (crosshairEl) {{ crosshairEl.remove(); crosshairEl = null; }}
}});

document.getElementById("toggle-optimized").addEventListener("change", (e) => {{ state.showOptimized = e.target.checked; render(); }});
document.getElementById("toggle-baseline").addEventListener("change", (e) => {{ state.showBaseline = e.target.checked; render(); }});
document.getElementById("toggle-ref").addEventListener("change", (e) => {{ state.showRef = e.target.checked; render(); }});
document.getElementById("btn-all").addEventListener("click", () => {{ state.rows = new Set(SERIES.map(s => s.row)); render(); }});
document.getElementById("btn-none").addEventListener("click", () => {{ state.rows = new Set(); render(); }});

if (window.matchMedia) {{
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
}}
new MutationObserver(render).observe(document.documentElement, {{ attributes: true, attributeFilter: ["data-theme"] }});

render();
</script>
"""


def render_html(series_payload: List[dict], row14_present: bool) -> str:
    ref = standard_normal_reference()
    row14_note = (
        "Row 14 (the S=100,000 chunked-attention shape) is measured in fp16 "
        "input / fp32 model, unlike rows 1-13 (fp32 throughout) -- its "
        "absolute ms scale is not comparable to the rest, but its normalized "
        "shape still is."
        if row14_present
        else ""
    )
    n_rows = len({s["row"] for s in series_payload})
    return PAGE_TEMPLATE.format(
        series_json=json.dumps(series_payload),
        ref_json=json.dumps([[round(x, 4), round(y, 6)] for x, y in ref]),
        z_range=Z_RANGE,
        bin_width=BIN_WIDTH,
        n_series=len(series_payload),
        n_rows=n_rows,
        row14_note=row14_note,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="latency_distribution_sweep.py JSON output file(s)")
    parser.add_argument("--out", default="results/artifacts/latency_distribution.html")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_series = load_series(args.inputs)
    if not raw_series:
        raise SystemExit("no series found in the given input file(s)")
    series_payload = build_series_payload(raw_series)
    row14_present = any(s["row"] == N_SHAPES for s in series_payload)
    html = render_html(series_payload, row14_present)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out} ({len(series_payload)} series)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
