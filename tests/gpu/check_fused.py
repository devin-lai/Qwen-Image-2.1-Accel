# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""GPU equivalence checks for the selective fusion kernels, including batch >1."""
import json
from pathlib import Path
import torch

from qwen_image_accel.kernels.elementwise import residual, rope_qk
from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen

torch.manual_seed(91)
results = []
for batch, seq in [(1, 49), (2, 257), (1, 4096)]:
    q = torch.randn(batch, seq, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    phase = torch.randn(seq, 64, device="cuda")
    freqs = torch.polar(torch.ones_like(phase), phase)
    x, y = rope_qk(q, k, freqs)
    rq = apply_rotary_emb_qwen(q, freqs, use_real=False)
    rk = apply_rotary_emb_qwen(k, freqs, use_real=False)
    h = q.flatten(2).contiguous()
    v = k.flatten(2).contiguous()
    gate = torch.randn(batch, 1, 4096, device="cuda", dtype=torch.bfloat16).tanh()
    r = residual(h, v, gate)
    ref = h + gate * v
    row = {"batch": batch, "seq": seq, "q_equal": torch.equal(x, rq),
           "k_equal": torch.equal(y, rk), "residual_equal": torch.equal(r, ref),
           "rope_max_abs": float((x-rq).abs().max()), "residual_max_abs": float((r-ref).abs().max())}
    results.append(row)
print(json.dumps(results, indent=2))
assert all(r["q_equal"] and r["k_equal"] and r["residual_equal"] for r in results)
