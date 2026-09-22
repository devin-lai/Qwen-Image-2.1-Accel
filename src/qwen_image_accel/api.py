# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Standalone generation and a small JSON adapter for Qwen-Image-2.1.

Init loads the pipeline once. Process returns a
(JSON result, status code, error message) tuple; status zero means success.
"""
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import threading
from urllib.request import urlopen

logger = logging.getLogger(__name__)
pipe = None
_inference_lock = threading.Lock()


def Init(config: str = ""):
    """Load local model weights and apply the selected optimization profile."""
    from .profiles import get_profile

    settings = json.loads(config) if config else {}
    profile = settings.get("profile", os.environ.get("QW21_PROFILE", "fused"))
    get_profile(profile)

    import torch
    from diffusers import QwenImage21Pipeline
    from qwen_image_accel.pipeline import configure_pipeline

    global pipe
    torch.set_num_threads(int(os.environ.get("QW21_CPU_THREADS", "8")))
    pipe = QwenImage21Pipeline.from_pretrained(
        settings.get("model", os.environ.get("QW21_MODEL_PATH", "models/Qwen-Image-2.1")),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    applied = configure_pipeline(pipe, profile)
    logger.info("Qwen optimization profile: %s", applied)
    if applied.get("approximate") or applied.get("compiled_decode_blocks"):
        logger.warning("This profile changes numerical results; see docs/optimization.md")


def algorithm_process(data: dict):
    """Generate an image from parameter fields and optional local/HTTP images."""
    import torch
    from PIL import Image

    if pipe is None:
        raise RuntimeError("Call Init() before generating images")
    param = data.get("parameter", {})
    images = []
    for media in data.get("media_info_list", []):
        source = media["media_data"]
        if source.startswith(("https://", "http://")):
            with urlopen(source, timeout=60) as response:
                image_data = BytesIO(response.read())
            with Image.open(image_data) as image:
                images.append(image.copy())
        else:
            with Image.open(source) as image:
                images.append(image.copy())
    kwargs = {
        "prompt": str(param.get("prompt", "")),
        "num_inference_steps": int(param.get("steps", 40)),
        "generator": torch.Generator("cuda").manual_seed(int(param.get("seed", 42))),
        "use_kv_cache": True,
    }
    if images:
        kwargs["image"] = images
    else:
        kwargs.update(width=int(param.get("width", 2048)), height=int(param.get("height", 2048)))
    output = Path(param.get("output", "outputs/output.png"))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Scheduler, offload hooks, and graph buffers are mutable per pipeline.
    with _inference_lock:
        image = pipe(**kwargs).images[0]
        image.save(output, format="PNG")
    return True, str(output)


def Process(body: str, extra: str = ""):
    """Handle JSON input using the standalone adapter's tuple response."""
    try:
        _, output = algorithm_process(json.loads(body))
    except Exception:
        logger.exception("Image generation failed")
        return json.dumps({}), 20301, "Image generation failed"
    result = {
        "parameter": {},
        "media_info_list": [{
            "media_data": output,
            "media_profiles": {"media_data_type": "png"},
            "media_extra": {},
        }],
        "extra": {},
    }
    return json.dumps(result), 0, ""
