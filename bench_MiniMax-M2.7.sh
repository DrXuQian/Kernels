#!/usr/bin/env bash
# Convenience alias for MiniMax-M2.7 TP=1.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT_DIR/bench_MiniMax-M2.7_TP1.sh" "$@"
