# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Run GPU regression checks serially on the validated runtime."""
import importlib.metadata
from pathlib import Path
import subprocess
import sys


def main():
    import torch
    if (not torch.cuda.is_available() or torch.__version__.split("+")[0] != "2.11.0"
            or torch.version.cuda != "13.0" or torch.cuda.get_device_capability() != (12, 0)
            or importlib.metadata.version("triton") != "3.6.0"):
        raise SystemExit("GPU checks require torch 2.11.0+cu130 / Triton 3.6.0 / SM120")
    for check in sorted(Path(__file__).parent.glob("check_*.py")):
        print(f"Running {check.name}", flush=True)
        subprocess.run([sys.executable, str(check)], check=True)


if __name__ == "__main__":
    main()
