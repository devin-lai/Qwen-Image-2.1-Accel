#!/usr/bin/env bash
# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
# Install into the active Python environment. Install CUDA/PyTorch separately.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
"$python_bin" -c 'import torch; print("Using PyTorch", torch.__version__)'
"$python_bin" "$project_dir/scripts/verify_assets.py" diffusers
"$python_bin" -m pip install "$project_dir/third_party/wheels/diffusers-0.41.0.dev0-py3-none-any.whl"
"$python_bin" -m pip install -e "$project_dir"
