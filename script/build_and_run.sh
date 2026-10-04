#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/cmake-build-metal"
MODE=run

case "${1:-}" in
  run|--debug|--logs|--telemetry|--verify|--build-only)
    MODE="$1"
    shift
    ;;
esac

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "The native Metal benchmark requires an Apple Silicon Mac." >&2
  exit 1
fi

cmake -S "$ROOT_DIR/metal" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DPYROWAVE_METAL_CLI=ON -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/output"
cmake --build "$BUILD_DIR" --parallel

APP_BINARY="$BUILD_DIR/pyrowave-metal-bench"
case "$MODE" in
  --build-only) ;;
  --debug) exec lldb -- "$APP_BINARY" "$@" ;;
  --verify) exec "$APP_BINARY" --frames 5 --warmup 2 "$@" ;;
  --logs|--telemetry)
    # This CLI reports configuration, timing, and errors directly to the terminal.
    exec "$APP_BINARY" "$@"
    ;;
  run) exec "$APP_BINARY" "$@" ;;
esac
