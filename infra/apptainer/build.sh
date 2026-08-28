#!/usr/bin/env bash
# Build the reproducible run image from infra/apptainer/kernel.def.
#
#   bash infra/apptainer/build.sh                 # -> /scratch/kernel.sif
#   bash infra/apptainer/build.sh ./kernel.sif    # custom output path
#
# Image contents (see kernel.def): nvidia/cuda:13.1.0-devel-ubuntu24.04,
# torch (cu13x wheel), triton, numpy/pandas, nsight-compute/systems 13-1.
# %environment pins TORCH_CUDA_ARCH_LIST=8.9 (sm_89 only -> fast JIT) and
# CUDA_MODULE_LOADING=LAZY.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEF="$ROOT/infra/apptainer/kernel.def"
SIF="${1:-/scratch/kernel.sif}"

command -v apptainer >/dev/null || { echo "build.sh: apptainer not on PATH" >&2; exit 1; }
[[ -f "$DEF" ]] || { echo "build.sh: definition not found: $DEF" >&2; exit 1; }

echo "build.sh: apptainer build $SIF $DEF"
apptainer build "$SIF" "$DEF"

cat <<'NOTE'

build.sh: image built. One host-side step is NOT run here (needs sudo + reboot):

    # ncu needs perf-counter access or every profile is empty
    sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
        > /etc/modprobe.d/nvidia-profiler.conf'
    sudo update-initramfs -u && sudo reboot

Clock locking belongs in the Slurm prolog (docs/SETUP.md 2), not in a job.
NOTE
