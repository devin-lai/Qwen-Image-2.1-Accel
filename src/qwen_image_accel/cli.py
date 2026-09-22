# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Generate or edit images from the command line."""
import argparse
import json
import logging
import os

from .api import Init, algorithm_process
from .profiles import PROFILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("QW21_MODEL_PATH", "models/Qwen-Image-2.1"))
    parser.add_argument("--profile", default=os.environ.get("QW21_PROFILE", "fused"), choices=tuple(PROFILES))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", action="append", default=[], help="Input image path or HTTP URL; repeat for multiple images")
    parser.add_argument("--output", default="outputs/output.png", help="Output PNG path")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    Init(json.dumps({"model": args.model, "profile": args.profile}))
    _, output = algorithm_process({
        "parameter": vars(args),
        "media_info_list": [{"media_data": source} for source in args.image],
    })
    print(output)


if __name__ == "__main__":
    main()
