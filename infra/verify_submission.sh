#!/usr/bin/env bash
# Extract a packaged tarball and smoke-test it as a grader would receive it.
#
#   bash infra/verify_submission.sh dist/techjam2_<ver>.tar.gz
#
# Checks: both entry points parse; the frozen baseline symbols and the CUDA
# triad are present; no build artefacts leaked into the archive; the generated
# entry point is in sync; and -- if a GPU + image are available -- that the
# custom kernels are actually discoverable from the repo root (the silent
# failure mode).
set -euo pipefail

TARBALL="${1:?usage: verify_submission.sh <tarball>}"
[[ -f "$TARBALL" ]] || { echo "not a file: $TARBALL" >&2; exit 1; }
TARBALL="$(cd "$(dirname "$TARBALL")" && pwd)/$(basename "$TARBALL")"

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
tar xzf "$TARBALL" -C "$T"
cd "$T/techjam2"
echo "verify_submission: extracted to $T/techjam2"
rc=0

# 1. both entry points parse
python3 -c "import ast; ast.parse(open('benchmark.py').read()); ast.parse(open('torch_transformer_benchmark.py').read())" \
  && echo "  [ok] benchmark.py + torch_transformer_benchmark.py parse" \
  || { echo "  [FAIL] entry point does not parse"; rc=1; }

# 2. frozen baseline symbols present in both
for sym in "class TransformerConfig" "class BaselineTransformer" "class UserOptimizedTransformer" \
           "def run_accuracy_tests" "def benchmark_models" "def main"; do
  grep -q "$sym" benchmark.py && grep -q "$sym" torch_transformer_benchmark.py \
    || { echo "  [FAIL] missing in an entry point: $sym"; rc=1; }
done
[[ $rc == 0 ]] && echo "  [ok] frozen baseline + optimized symbols present in both"

# 3. CUDA triad present
for f in csrc/cublaslt_algo.cpp csrc/g4_4_warpspec_gemm.cpp csrc/g4_4_warpspec_gemm.cu; do
  [[ -f "$f" ]] || { echo "  [FAIL] missing runtime CUDA source: $f"; rc=1; }
done
[[ $rc == 0 ]] && echo "  [ok] csrc runtime triad present"

# 4. no build artefacts leaked
if tar tzf "$TARBALL" | grep -Eq '\.(so|o|cubin|ncu-rep|nsys-rep)$|/__pycache__/|/\.ext_build/|/\.cutlass/|/\.sass/'; then
  echo "  [FAIL] archive contains build artefacts:"; tar tzf "$TARBALL" | grep -E '\.(so|o|cubin)$|__pycache__|\.ext_build/' | head
  rc=1
else
  echo "  [ok] archive is clean (no .so/.o/.cubin/caches)"
fi

# 5. generated entry point in sync (needs the judges' canonical)
if [[ -f "$HOME/torch_transformer_benchmark.py" ]]; then
  python3 tools/sync_entrypoint.py --check && echo "  [ok] torch_transformer_benchmark.py in sync" \
    || { echo "  [FAIL] torch_transformer_benchmark.py stale in the tarball"; rc=1; }
else
  echo "  [skip] ~/torch_transformer_benchmark.py absent -- cannot check entrypoint sync"
fi

# 6. kernels discoverable from repo root (only if GPU + image)
SIF="${KERNEL_SIF:-/scratch/kernel.sif}"
if command -v apptainer >/dev/null && [[ -e "$SIF" ]] && command -v nvidia-smi >/dev/null; then
  apptainer exec --nv --cleanenv --bind "$T/techjam2":/work "$SIF" python3 -c "
import sys; sys.path.insert(0,'/work'); import benchmark
lt = benchmark._lt_ext(); ws = benchmark._ws_ext()
print('  [%s] custom kernels: lt=%s ws=%s' % ('ok' if (lt is not None and ws is not None) else 'FAIL',
      lt is not None, ws is not None))
assert lt is not None and ws is not None, 'KERNEL DISENGAGED -- csrc/ not reachable from benchmark.py'
" || rc=1
else
  echo "  [skip] no GPU/image -- cannot verify kernels engage"
fi

echo
[[ $rc == 0 ]] && echo "verify_submission: PASS" || echo "verify_submission: FAIL"
exit $rc
