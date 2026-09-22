#!/usr/bin/env bash
# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
# Optional, pinned SageAttention2 build into the active Python environment.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
"$python_bin" "$project_dir/scripts/verify_assets.py" sageattention
"$python_bin" -c 'import torch; assert torch.__version__.startswith("2.11.0") and torch.version.cuda == "13.0"; assert torch.cuda.get_device_capability() == (12, 0)'
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/qw21-sage.XXXXXX")"
trap 'rm -rf "$build_dir"' EXIT
archive="$project_dir/third_party/sources/sageattention-d1a57a5.tar.gz"
tar -xzf "$archive" -C "$build_dir"
MAX_JOBS="${MAX_JOBS:-8}" TORCH_CUDA_ARCH_LIST=12.0 \
  "$python_bin" -m pip install --no-build-isolation --no-deps \
  "$build_dir/SageAttention-d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
