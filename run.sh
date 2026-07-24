#!/usr/bin/env bash
# Start Moshi's PyTorch server with the experimental 8-bit checkpoint.
set -euo pipefail

cd "$(dirname "$0")"
# Do not depend on the optional hf_transfer package when a parent shell enables it.
exec env HF_HUB_ENABLE_HF_TRANSFER=0 .venv/bin/python -m moshi.server \
  --hf-repo kyutai/moshika-pytorch-q8 \
  --half \
  "$@"
