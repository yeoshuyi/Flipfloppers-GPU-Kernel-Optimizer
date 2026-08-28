#!/usr/bin/env bash
# Single source of truth for the Apptainer invocation used everywhere in this
# repo. Everything else (run_eval.sh, ad-hoc probes) should call this so the
# bind mounts and image path are defined in exactly one place.
#
#   infra/run_container.sh python3 /work/torch_transformer_benchmark.py --causal ...
#   infra/run_container.sh python3 /work/experiments/final_scorecard.py
#
# --bind "$ROOT":/work is what makes ./csrc resolve to /work/csrc inside the
# container -- the same layout the judges' harness expects.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${KERNEL_SIF:-/scratch/kernel.sif}"

if [[ ! -e "$SIF" ]]; then
  echo "infra/run_container.sh: image not found: $SIF" >&2
  echo "  build it with:  bash infra/apptainer/build.sh" >&2
  exit 1
fi

BINDS=(--bind "$ROOT":/work)
# Optional: expose the Slurm run-output dir if a caller wants /runs.
if [[ -n "${RUNS_BIND:-}" ]]; then
  BINDS+=(--bind "${RUNS_BIND}":/runs)
fi

exec apptainer exec --nv --cleanenv "${BINDS[@]}" "$SIF" "$@"
