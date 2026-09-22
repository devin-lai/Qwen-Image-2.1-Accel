# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Verify the pinned runtime inputs before installation (standard library only)."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", nargs="?", choices=["diffusers", "sageattention"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "third_party"
    manifest = json.loads((root / "manifest.json").read_text())
    for name in ([args.asset] if args.asset else manifest):
        item = manifest[name]
        path = root / item["path"]
        if not path.is_file():
            raise SystemExit(f"Missing {path}; obtain the pinned asset from the project checkout")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit(f"Checksum mismatch: {path}; restore the original pinned asset")
        print(f"Verified {name}: {item['path']}")


if __name__ == "__main__":
    main()
