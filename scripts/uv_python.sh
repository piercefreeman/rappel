#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../python" && pwd)"
uv_project="${UV_PROJECT:-$project_dir}"

exec uv run --project "$uv_project" python "$@"
