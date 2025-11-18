#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE=${1:-}
if [[ "$BUILD_TYPE" != "develop" && "$BUILD_TYPE" != "release" ]]; then
  echo "Usage: $0 <develop|release> [extra maturin args]"
  exit 1
fi
shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$REPO_ROOT/python"
TMPDIR_NAME=".merge-tmp"
TMPDIR="$REPO_ROOT/$TMPDIR_NAME"
MATURIN_DEVELOP="uv run maturin develop --release --uv"
MATURIN_BUILD="uv run maturin build --release"

rm -rf "$TMPDIR"
mkdir -p "$TMPDIR/tmp1" "$TMPDIR/tmp2"

if [[ "$BUILD_TYPE" == "develop" ]]; then
  (cd "$PYTHON_DIR" && $MATURIN_DEVELOP --bindings pyo3 "$@")
  (cd "$PYTHON_DIR" && $MATURIN_DEVELOP --bindings bin "$@")
  echo "Development build completed."
  exit 0
fi

(cd "$PYTHON_DIR" && $MATURIN_BUILD --bindings pyo3 -o "$TMPDIR/tmp1" "$@")
(cd "$PYTHON_DIR" && $MATURIN_BUILD --bindings bin -o "$TMPDIR/tmp2" "$@")

"$REPO_ROOT/merge_wheels.sh" "$TMPDIR_NAME" "target/wheels"
rm -rf "$TMPDIR"
echo "Release wheel written to target/wheels"
