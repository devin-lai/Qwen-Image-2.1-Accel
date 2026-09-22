# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Measure a profile's numerical error along the *baseline's* denoising path.

Image-to-image PSNR between two full samples conflates two different things: how
much arithmetic error a profile introduces, and how much a 40-step sampler
amplifies any perturbation. A prompt whose fine detail sits near a decision
boundary can diverge visibly from an error far too small to call degradation.

This runs the baseline once, records the latent entering every step, then replays
exactly those latents through the candidate and compares the velocity prediction
step by step. The result is the model's own error, with the sampler's feedback
loop removed.
"""
import argparse, json, os
from pathlib import Path



from common import PROMPTS
from qwen_image_accel import PROFILES, configure_pipeline


def build(profile, model):
    pipe = QwenImage21Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16,
                                               local_files_only=True)
    return pipe, configure_pipeline(pipe, profile)


def trace(pipe, prompt, size, steps, replay=None):
    """Return (velocity predictions, latents after each step), optionally replayed."""
    predictions, produced, index = [], [], [0]
    forward, step = pipe.transformer.forward, pipe.scheduler.step

    def traced_forward(*args, **kwargs):
        out = forward(*args, **kwargs)
        predictions.append(out[0].detach().float().cpu())
        return out

    def traced_step(*args, **kwargs):
        out = step(*args, **kwargs)
        i = index[0]; index[0] += 1
        if replay is None:
            produced.append((out[0] if isinstance(out, tuple) else out.prev_sample).detach().cpu())
            return out
        forced = replay[i].to("cuda", torch.bfloat16)
        if isinstance(out, tuple):
            return (forced,) + tuple(out[1:])
        out.prev_sample = forced
        return out

    pipe.transformer.forward, pipe.scheduler.step = traced_forward, traced_step
    try:
        pipe(prompt=prompt, width=size, height=size, num_inference_steps=steps,
             generator=torch.Generator("cuda").manual_seed(42), output_type="latent")
        torch.cuda.synchronize()
    finally:
        pipe.transformer.forward, pipe.scheduler.step = forward, step
    return predictions, produced


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", nargs="+", required=True, choices=tuple(PROFILES))
    p.add_argument("--model", default=os.environ.get("QW21_MODEL_PATH", "models/Qwen-Image-2.1"))
    p.add_argument("--sizes", type=int, nargs="+", default=[1024])
    p.add_argument("--prompts", nargs="+", choices=tuple(PROMPTS), default=["product", "type"])
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.steps < 1 or any(size < 1 for size in a.sizes):
        p.error("steps and sizes must be positive")
    global torch, QwenImage21Pipeline
    import torch
    from diffusers import QwenImage21Pipeline
    torch.set_num_threads(8)

    reference = {}
    pipe, _ = build("baseline", a.model)
    for size in a.sizes:
        for key in a.prompts:
            predictions, latents = trace(pipe, PROMPTS[key], size, a.steps)
            reference[(size, key)] = (predictions, latents)
    del pipe; torch.cuda.empty_cache()

    report = {"steps": a.steps, "profiles": {}}
    for profile in a.profiles:
        pipe, config = build(profile, a.model)
        entry = {"config": {k: v for k, v in config.items() if not isinstance(v, (list, dict))}, "cases": {}}
        for size in a.sizes:
            for key in a.prompts:
                base_pred, base_latents = reference[(size, key)]
                predictions, _ = trace(pipe, PROMPTS[key], size, a.steps, replay=base_latents)
                errors = []
                for reference_pred, candidate in zip(base_pred, predictions):
                    delta = (candidate - reference_pred).pow(2).mean().sqrt()
                    errors.append(float(delta / reference_pred.pow(2).mean().sqrt()))
                entry["cases"][f"{key}_{size}"] = {
                    "relative_rmse_mean": sum(errors) / len(errors),
                    "relative_rmse_max": max(errors),
                    "relative_rmse_first": errors[0],
                    "relative_rmse_last": errors[-1],
                    "per_step": [round(e, 6) for e in errors]}
        report["profiles"][profile] = entry
        del pipe; torch.cuda.empty_cache()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=1))
    print("QW_JSON_START")
    print(json.dumps({p: {k: {m: round(v[m], 5) for m in ("relative_rmse_mean", "relative_rmse_max",
                                                          "relative_rmse_first", "relative_rmse_last")}
                          for k, v in e["cases"].items()} for p, e in report["profiles"].items()}, indent=1))
    print("QW_JSON_END")


if __name__ == "__main__":
    main()
