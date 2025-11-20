#!/usr/bin/env bash
set -euo pipefail

HTTP_PORT="${CARABINER_HTTP_PORT:-24117}"
GRPC_PORT="${CARABINER_GRPC_PORT:-24118}"

rappel-server --http-addr "0.0.0.0:${HTTP_PORT}" --grpc-addr "0.0.0.0:${GRPC_PORT}" &
server_pid=$!

cleanup() {
  echo "Shutting down rappel daemons" >&2 || true
  if kill -0 "${server_pid}" >/dev/null 2>&1; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" || true
  fi
}

trap cleanup EXIT

start_workers
