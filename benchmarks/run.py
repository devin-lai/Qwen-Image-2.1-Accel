# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Benchmark fixed prompts at one or more resolutions, with per-size warmup."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from common import PROMPTS, StageTimer
from qwen_image_accel import PROFILES, configure_pipeline


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("QW21_MODEL_PATH", "models/Qwen-Image-2.1"))
    parser.add_argument("--profile", choices=tuple(PROFILES), default="fused")
    parser.add_argument("--sizes", nargs="+", type=positive_int, default=[512, 1024, 2048])
    parser.add_argument("--steps", type=positive_int, default=40)
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=positive_int, default=2)
    parser.add_argument("--quality", action="store_true", help="Also run portrait and typography prompts once per size")
    parser.add_argument("--image", help="Local reference image for editing; input dimensions take precedence")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from diffusers import QwenImage21Pipeline
    from PIL import Image

    torch.set_num_threads(int(os.environ.get("QW21_CPU_THREADS", "8")))
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True)
    config = configure_pipeline(pipe, args.profile)
    torch.cuda.synchronize()
    report = dict(args=vars(args), config=config, load_seconds=time.perf_counter()-start,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  packages={n: importlib.metadata.version(n) for n in ("diffusers", "transformers", "triton")},
                  warmups=[], runs=[])
    report["implementation_hashes"] = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in tuple(sys.modules.items())
        if name.startswith("qwen_image_accel") and getattr(module, "__file__", None)
    }
    timer = StageTimer()
    timer.wrap(pipe, "encode_prompt", "text_encode")
    timer.wrap(pipe.transformer, "forward", "dit")
    timer.wrap(pipe.vae, "decode", "vae")
    image = None
    if args.image:
        with Image.open(args.image) as source:
            image = source.copy()
    sizes = [None] if image is not None else args.sizes
    for size in sizes:
        dimensions = {"image": [image]} if image is not None else {"width": size, "height": size}
        prompt_key = "edit" if image is not None else "product"
        start = time.perf_counter()
        pipe(prompt=PROMPTS[prompt_key], num_inference_steps=args.warmup_steps, use_kv_cache=True,
             generator=torch.Generator("cuda").manual_seed(args.seed), **dimensions)
        torch.cuda.synchronize()
        report["warmups"].append(dict(size=size, seconds=time.perf_counter()-start, stages=timer.collect()))
        cases = [(prompt_key, repeat) for repeat in range(args.repeats)]
        if args.quality and image is None:
            cases += [("portrait", 0), ("type", 0)]
        for key, repeat in cases:
            final_latent = []

            def callback(pipeline, index, timestep, kwargs):
                if index == args.steps - 1:
                    final_latent.append(kwargs["latents"].detach())
                return kwargs

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = pipe(prompt=PROMPTS[key], num_inference_steps=args.steps, use_kv_cache=True,
                          generator=torch.Generator("cuda").manual_seed(args.seed),
                          callback_on_step_end=callback, **dimensions).images[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            name = f"{key}_seed{args.seed}_{size or 'edit'}_{args.steps}_r{repeat}"
            row = dict(name=name, size=size, image_size=list(result.size), prompt=PROMPTS[key],
                       seconds=elapsed, stages_seconds=timer.collect(),
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            result.save(output / f"{name}.png")
            torch.save(final_latent[0].cpu(), output / f"{name}.pt")
            report["runs"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print("RESULT", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
