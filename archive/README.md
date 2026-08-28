# `archive/` — MAP-Elites elite-config store

A small 2D archive of the best result found per **regime × precision family**.
It replaced a would-be "archivist" agent with zero token cost — `tools/archive.py`
reads and writes it via a relative `Path("archive")`, so this directory must stay
a direct child of the repo root and `archive.py` must be run from there.

## Files

`<regime>[__family].json`, one cell each:

- **regime** ∈ `default`, `tiny`, `padded`, `large-batch`, `long-seq`, and their
  `causal-*` variants.
- **family** ∈ *(none)* = the project's original TF32/FP32 budget,
  `__trackA` = the TF32-only safety track, `__fp16` = the FP16-storage track.

Each file holds `{"elite": {...}, "log": [...]}` where an entry is
`{id, speedup, applied: [G-stage ids], ts}`. `applied` is the exact optimization
stack that produced that speedup — the audit trail behind every number in
`docs/DOCUMENTATION.md` §3 and `SUBMISSION.md`.

Current shipped causal elites (`causal-*__fp16.json`): default 2.71×, tiny 7.66×,
long-seq 7.78×, large-batch 2.98×.

## Usage

```bash
python3 tools/archive.py query causal-large-batch          # inspect a cell
python3 tools/archive.py commit <cell> <id> <speedup> ...  # record a new elite (+ git commit)
```
