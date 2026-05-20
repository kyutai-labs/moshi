#!/usr/bin/env bash
# Human-in-the-loop Ralph: runs exactly one iteration
# Use this to observe behavior before going fully autonomous

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export MAX_ITERATIONS=1
exec "$SCRIPT_DIR/ralph.sh" "$@"
