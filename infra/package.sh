#!/usr/bin/env bash
# Produce a clean submission tarball from the committed tree.
#
#   bash infra/package.sh                 # dist/techjam2_<ver>.tar.gz
#   bash infra/package.sh --allow-dirty   # skip the clean-tree check
#
# git archive already omits untracked files and the gitignored trees
# (.ext_build/ ~300MB, .cutlass/ 181MB, .sass/ 91MB, __pycache__/). The
# 120 slurm receipts under results/logs/ are kept on purpose -- they back
# every number in the docs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ALLOW_DIRTY=0
[[ "${1:-}" == "--allow-dirty" ]] && ALLOW_DIRTY=1

# --- clean tree ---------------------------------------------------------
if [[ "$ALLOW_DIRTY" == "0" && -n "$(git status --porcelain)" ]]; then
  echo "package.sh: working tree not clean -- commit first, or pass --allow-dirty" >&2
  git status --short >&2
  exit 1
fi

# --- guards -----------------------------------------------------------
fail=0
if [[ -f "$HOME/torch_transformer_benchmark.py" ]]; then
  python3 tools/verify_baseline.py       || fail=1
  python3 tools/sync_entrypoint.py --check || fail=1
else
  echo "package.sh: WARNING ~/torch_transformer_benchmark.py absent -- skipping baseline/entrypoint checks"
fi
python3 tools/check_validity.py torch_transformer_benchmark.py || fail=1
[[ "$fail" == "0" ]] || { echo "package.sh: guard(s) failed" >&2; exit 1; }

# --- triad sanity ---------------------------------------------------
for f in benchmark.py torch_transformer_benchmark.py \
         csrc/cublaslt_algo.cpp csrc/g4_4_warpspec_gemm.cpp csrc/g4_4_warpspec_gemm.cu; do
  [[ -f "$f" ]] || { echo "package.sh: missing frozen-location file: $f" >&2; exit 1; }
done

# --- archive ------------------------------------------------------
VER="$(git describe --always --dirty --tags 2>/dev/null || git rev-parse --short HEAD)"
mkdir -p dist
OUT="dist/techjam2_${VER}.tar.gz"
git archive --format=tar.gz --prefix=techjam2/ -o "$OUT" HEAD

echo "package.sh: wrote $OUT"
echo "  $(du -h "$OUT" | cut -f1)   $(tar tzf "$OUT" | wc -l) entries"
sha256sum "$OUT"
echo "--- directories + root files ---"
tar tzf "$OUT" | sed 's#^techjam2/##' | grep -E '/$|^[^/]+$' | sort -u
echo
echo "verify with:  bash infra/verify_submission.sh $OUT"
