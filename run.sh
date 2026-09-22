#!/usr/bin/env bash
# Generate or edit an image with the standalone Python entry point.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m qwen_image_accel "$@"
