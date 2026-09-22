# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Validate the fused MXFP8 epilogues against the separate kernels they replace."""
import json, time
from pathlib import Path
import torch


from qwen_image_accel.kernels import mxfp8
from qwen_image_accel.kernels.bf16 import swiglu
from qwen_image_accel.kernels.layernorm import layer_norm_scale
from qwen_image_accel.kernels.epilogue import layer_norm_scale_quantize, swiglu_quantize

dev = "cuda"
res = {"layernorm": [], "swiglu": [], "speed": {}}

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1e3

def same(a, b):
    return (torch.equal(a.values.view(torch.uint8), b.values.view(torch.uint8))
            and torch.equal(a.scales.view(torch.uint8), b.scales.view(torch.uint8))
            and a.rows == b.rows)

torch.manual_seed(0)
norm = torch.nn.LayerNorm(4096, elementwise_affine=False, eps=1e-6).cuda()
for batch, seq in [(1, 1024), (1, 1536), (1, 2400), (1, 4096), (1, 16384), (2, 1024), (1, 77)]:
    x = torch.randn(batch, seq, 4096, device=dev, dtype=torch.bfloat16)
    x[..., ::512] *= 25
    scale = torch.randn(batch, 1, 4096, device=dev, dtype=torch.bfloat16)
    fused = layer_norm_scale_quantize(x, scale, norm)
    reference = mxfp8.quantize(layer_norm_scale(x, scale, norm).reshape(-1, 4096))
    res["layernorm"].append({"batch": batch, "seq": seq, "identical_to_separate": bool(same(fused, reference)),
                             "shape_preserved": tuple(fused.shape) == tuple(x.shape)})
    del x, scale, fused, reference; torch.cuda.empty_cache()

for rows, width in [(1024, 12288), (2400, 12288), (16384, 12288), (128, 12288)]:
    g = torch.randn(rows, width, device=dev, dtype=torch.bfloat16)
    v = torch.randn(rows, width, device=dev, dtype=torch.bfloat16)
    fused = swiglu_quantize(g, v)
    reference = mxfp8.quantize(swiglu(g, v))
    res["swiglu"].append({"rows": rows, "width": width, "identical_to_separate": bool(same(fused, reference))})
    del g, v, fused, reference; torch.cuda.empty_cache()

for rows in (1024, 4096, 16384):
    x = torch.randn(1, rows, 4096, device=dev, dtype=torch.bfloat16)
    scale = torch.randn(1, 1, 4096, device=dev, dtype=torch.bfloat16)
    separate = bench(lambda: mxfp8.quantize(layer_norm_scale(x, scale, norm).reshape(-1, 4096)))
    fused = bench(lambda: layer_norm_scale_quantize(x, scale, norm))
    res["speed"][f"layernorm_M{rows}"] = {"separate_ms": round(separate, 4), "fused_ms": round(fused, 4),
                                          "saved_ms": round(separate - fused, 4)}
    g = torch.randn(rows, 12288, device=dev, dtype=torch.bfloat16)
    v = torch.randn(rows, 12288, device=dev, dtype=torch.bfloat16)
    separate = bench(lambda: mxfp8.quantize(swiglu(g, v)))
    fused = bench(lambda: swiglu_quantize(g, v))
    res["speed"][f"swiglu_M{rows}"] = {"separate_ms": round(separate, 4), "fused_ms": round(fused, 4),
                                       "saved_ms": round(separate - fused, 4)}
    del x, scale, g, v; torch.cuda.empty_cache()

res["all_identical"] = all(r["identical_to_separate"] for r in res["layernorm"] + res["swiglu"])
print("QW_JSON_START"); print(json.dumps(res, indent=1)); print("QW_JSON_END")

assert res["all_identical"] and all(r["shape_preserved"] for r in res["layernorm"]), res
