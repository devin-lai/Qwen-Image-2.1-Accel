# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Validate the MXFP8 Triton quantizer against a reference and measure speed."""
import json, time
from pathlib import Path
import torch


from qwen_image_accel.optimizations import mxfp8

dev = "cuda"
res = {"checks": [], "speed": {}}

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters

def ref_quant(x):
    """Reference: per-32 block, exponent = smallest power of two without clipping."""
    m, k = x.shape
    xv = x.float().view(m, k // 32, 32)
    amax = xv.abs().amax(-1)
    bits = amax.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 8 + ((bits & 0x7FFFFF) > 6291456).int()
    e = torch.where(amax == 0, torch.zeros_like(e), e.clamp(0, 254))
    scale = torch.exp2((e - 127).float())
    q = (xv * torch.exp2((127 - e).float())[..., None]).to(torch.float8_e4m3fn)
    return q.view(m, k), e.to(torch.uint8), scale

def swizzle(bits):
    m, kb = bits.shape
    return bits.view(m // 128, 4, 32, kb // 4, 4).permute(0, 3, 2, 1, 4).contiguous().view(-1)

torch.manual_seed(0)
# --- kernel equivalence, including outliers, padding and non-multiple rows ---
for m, k in [(128, 4096), (1024, 4096), (1536, 4096), (2400, 4096), (4096, 12288), (16384, 4096), (77, 4096), (1, 4096)]:
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    x[:, ::512] *= 40
    if m > 200: x[m // 3] *= 1e-3
    a = mxfp8.quantize(x)
    q, s, rows = a.values, a.scales, a.rows
    padded = q.shape[0]
    xp = torch.zeros(padded, k, device=dev, dtype=torch.bfloat16); xp[:m] = x
    rq, re, _ = ref_quant(xp)
    ok_q = torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    ok_s = torch.equal(s.view(torch.uint8), swizzle(re))
    res["checks"].append({"m": m, "k": k, "rows": rows, "padded": padded,
                          "fp8_exact": bool(ok_q), "scale_layout_exact": bool(ok_s)})

# --- end-to-end linear numerics vs BF16 ---
acc = {}
for k, n, label in [(4096, 4096, "out"), (4096, 12288, "qkv"), (12288, 4096, "mlp_out")]:
    torch.manual_seed(1)
    lin = torch.nn.Linear(k, n, bias=False, device=dev, dtype=torch.bfloat16)
    x = torch.randn(4096, k, device=dev, dtype=torch.bfloat16); x[:, ::256] *= 12
    ref = (x.float() @ lin.weight.float().t())
    rms = ref.pow(2).mean().sqrt()
    def nrmse(y): return ((y.float() - ref).pow(2).mean().sqrt() / rms).item()
    mx = mxfp8.MXFP8Linear(lin)
    acc[label] = {"bf16": nrmse(lin(x)), "mxfp8": nrmse(mx(x))}
    del lin, mx; torch.cuda.empty_cache()
res["linear_nrmse"] = acc

# --- speed incl. quantization at the three resolutions' token counts ---
for m in (1024, 4096, 16384):
    x = torch.randn(m, 4096, device=dev, dtype=torch.bfloat16)
    res["speed"][f"quantize_M{m}_K4096_ms"] = bench(lambda: mxfp8.quantize(x)) * 1e3
    for k, n, label in [(4096, 4096, "out"), (4096, 12288, "qkv"), (4096, 12288, "mlp_gate"), (12288, 4096, "mlp_out")]:
        xx = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        lin = torch.nn.Linear(k, n, bias=False, device=dev, dtype=torch.bfloat16)
        mx = mxfp8.MXFP8Linear(lin)
        t_bf = bench(lambda: lin(xx)); t_mx = bench(lambda: mx(xx))
        activation = mxfp8.quantize(xx)
        t_gemm = bench(lambda: mx.matmul(activation))
        res["speed"][f"{label}_M{m}"] = {"bf16_ms": round(t_bf*1e3, 3), "mxfp8_ms": round(t_mx*1e3, 3),
                                         "mxfp8_gemm_only_ms": round(t_gemm*1e3, 3),
                                         "speedup": round(t_bf/t_mx, 2), "speedup_shared_quant": round(t_bf/t_gemm, 2)}
        del lin, mx, xx, activation; torch.cuda.empty_cache()
    del x; torch.cuda.empty_cache()

res["all_checks_pass"] = all(c["fp8_exact"] and c["scale_layout_exact"] for c in res["checks"])
print("QW_JSON_START"); print(json.dumps(res, indent=1)); print("QW_JSON_END")

assert res["all_checks_pass"], res["checks"]
