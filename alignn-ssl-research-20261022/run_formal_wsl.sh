#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Use the activated environment, or an explicitly selected interpreter.
ALIGNN_RUNTIME_PYTHON="${ALIGNN_PYTHON:-python}"
if ! command -v "$ALIGNN_RUNTIME_PYTHON" >/dev/null 2>&1; then
  echo "Python not found. Activate your environment or set ALIGNN_PYTHON." >&2
  exit 2
fi
exec "$ALIGNN_RUNTIME_PYTHON" -u run_formal_display.py --single-process-loader "$@"
