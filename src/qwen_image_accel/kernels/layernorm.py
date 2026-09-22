# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""LayerNorm + timestep scale, preserving PyTorch 2.11 CUDA reduction order.

For the validated 4096-wide BF16 path: 128 virtual threads, four adjacent
values per vector, 32 online updates per thread, then native warp/block merges.
Reference: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu
The generic two-pass variance formula changes BF16 outputs and is not used here.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _welford_scale(
    X, S, O, SEQ, D: tl.constexpr, EPS: tl.constexpr, SINGLE_BATCH: tl.constexpr
):
    row = tl.program_id(0)
    lane = tl.arange(0, 128)
    mean = tl.full((128,), 0, tl.float32)
    var = tl.full((128,), 0, tl.float32)
    # Match the native vectorized CUDA kernel's input order and explicit FMAs.
    for i in tl.static_range(32):
        offset = lane * 4 + i % 4 + (i // 4) * 512
        x = tl.load(X + row * D + offset).to(tl.float32)
        delta = x - mean
        mean = libdevice.fma(delta, 1.0 / (i + 1), mean)
        var = libdevice.fma(delta, x - mean, var)
    for step in tl.static_range(7):
        # Five shuffle-down warp merges, then warp 0+2 and warp 0+1.
        if step < 5:
            offset = 16 >> step
            other = (lane // 32) * 32 + ((lane + offset) % 32)
        elif step == 5:
            other = (lane + 64) % 128
        else:
            other = (lane + 32) % 128
        m2 = tl.gather(mean, other, 0)
        v2 = tl.gather(var, other, 0)
        delta = mean - m2
        mean = libdevice.fma(m2, 0.5, mean * 0.5)
        var = libdevice.fma(delta * delta * (32 << step), 0.5, v2 + var)
    m = tl.sum(tl.where(lane == 0, mean, 0), 0)
    variance = tl.sum(tl.where(lane == 0, var, 0), 0) / D
    col = tl.arange(0, D)
    x = tl.load(X + row * D + col).to(tl.float32)
    y = ((x - m) * tl.rsqrt(variance + EPS)).to(tl.bfloat16).to(tl.float32)
    si = col
    if not SINGLE_BATCH:
        si = (row // SEQ) * D + col
    s = tl.load(S + si).to(tl.float32)
    tl.store(O + row * D + col, y * s)


def layer_norm_scale(x, scale, norm):
    if (
        x.ndim != 3
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.shape[-1] != 4096
        or x.numel() == 0
        or scale.shape != (x.shape[0], 1, 4096)
        or scale.dtype != torch.bfloat16
        or scale.device != x.device
        or not scale.is_contiguous()
        or norm.elementwise_affine
        or tuple(norm.normalized_shape) != (4096,)
    ):
        return norm(x) * scale
    out = torch.empty_like(x)
    _welford_scale[(x.numel() // 4096,)](
        x,
        scale,
        out,
        x.shape[1],
        D=4096,
        EPS=norm.eps,
        SINGLE_BATCH=x.shape[0] == 1,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
