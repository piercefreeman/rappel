#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <input_dir> <output_dir>"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$REPO_ROOT/$1"
OUTPUT_DIR="$REPO_ROOT/$2"
TMPDIR="$REPO_ROOT/.merged-wheel"

mkdir -p "$OUTPUT_DIR"

cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

merge_wheel() {
  local file_name="$1"
  rm -rf "$TMPDIR"
  mkdir -p "$TMPDIR"
  unzip -qo "$INPUT_DIR/tmp1/$file_name" -d "$TMPDIR"
  unzip -qo "$INPUT_DIR/tmp2/$file_name" -d "$TMPDIR"

  unzip -qjo "$INPUT_DIR/tmp1/$file_name" "*.dist-info/RECORD" -d "$INPUT_DIR/tmp1"
  unzip -qjo "$INPUT_DIR/tmp2/$file_name" "*.dist-info/RECORD" -d "$INPUT_DIR/tmp2"
  local record_path
  record_path=$(find "$TMPDIR" -path "*.dist-info" -type d | head -n1)/RECORD
  cat "$INPUT_DIR/tmp1/RECORD" "$INPUT_DIR/tmp2/RECORD" | sort | uniq > "$record_path"
  rm -f "$INPUT_DIR/tmp1/RECORD" "$INPUT_DIR/tmp2/RECORD"

  (cd "$TMPDIR" && zip -qr "$OUTPUT_DIR/$file_name" .)
}

for wheel in $(find "$INPUT_DIR/tmp1" -name "*.whl" -exec basename {} \; | sort -u); do
  if [[ ! -f "$INPUT_DIR/tmp2/$wheel" ]]; then
    echo "Skipping $wheel because it is missing from tmp2"
    continue
  fi
  merge_wheel "$wheel"
done
echo "Merged wheels available under $OUTPUT_DIR"
