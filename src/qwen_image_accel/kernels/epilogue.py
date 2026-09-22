# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Fused MXFP8 epilogues: produce FP8 directly instead of BF16 then FP8.

Every GEMM input on the quantized decode path is the output of a pointwise
kernel, so materializing it in BF16 costs a full write and a full read that the
FP8 tensor never needs. At 2048x2048 the normalized activation is 134 MiB and
the SwiGLU result is 402 MiB per block, so those round trips are the largest
remaining memory traffic after the GEMMs themselves.

Both kernels keep the arithmetic of their BF16 counterparts -- the native-order
Welford reduction and the explicit PTX SiLU -- and only replace the store, so a
profile differs from the BF16 default exactly by its FP8 GEMMs and nothing else.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from qwen_image_accel.kernels.mxfp8 import Quantized, padded_rows


@triton.jit
def _welford_scale_quantize(X, SCALE, Q, S, M, SEQ, KB_TILES,
                            D: tl.constexpr, EPS: tl.constexpr, SINGLE_BATCH: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 128)
    # Rows past the end exist only to fill the last 128-row scale tile. They load
    # nothing, quantize to zero, and contribute nothing to the padded GEMM.
    live = lane < 128 if row < M else lane < 0
    mean = tl.full((128,), 0, tl.float32)
    var = tl.full((128,), 0, tl.float32)
    # Native vectorized CUDA order: four adjacent values per virtual thread.
    for i in tl.static_range(32):
        offset = lane * 4 + i % 4 + (i // 4) * 512
        x = tl.load(X + row * D + offset, mask=live, other=0.0).to(tl.float32)
        delta = x - mean
        mean = libdevice.fma(delta, 1.0 / (i + 1), mean)
        var = libdevice.fma(delta, x - mean, var)
    for step in tl.static_range(7):
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
    wide = col < D if row < M else col < 0
    x = tl.load(X + row * D + col, mask=wide, other=0.0).to(tl.float32)
    y = ((x - m) * tl.rsqrt(variance + EPS)).to(tl.bfloat16).to(tl.float32)
    index = col if SINGLE_BATCH else (row // SEQ) * D + col
    scaled = tl.where(wide, y * tl.load(SCALE + index, mask=wide, other=0.0).to(tl.float32), 0.0)
    # The unfused path stores this in BF16 before quantizing. Keeping that
    # rounding means a quantized profile differs from the BF16 default by its
    # FP8 GEMMs alone, which is what the equivalence test asserts.
    blocks = tl.reshape(scaled.to(tl.bfloat16).to(tl.float32), (D // 32, 32))
    amax = tl.max(tl.abs(blocks), axis=1)
    bits = amax.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) - 8 + tl.where((bits & 0x7FFFFF) > 6291456, 1, 0)
    exponent = tl.where(amax == 0.0, 0, tl.maximum(tl.minimum(exponent, 254), 0))
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    tl.store(Q + row * D + col, tl.reshape(blocks * inverse[:, None], (D,)).to(Q.dtype.element_ty))
    kb = tl.arange(0, D // 32)
    tl.store(S + ((row // 128) * KB_TILES + kb // 4) * 512
             + (row % 32) * 16 + ((row % 128) // 32) * 4 + kb % 4, exponent.to(tl.uint8))


def layer_norm_scale_quantize(x, scale, norm):
    """LayerNorm, timestep scale and MXFP8 quantization in one pass, or None."""
    if (x.ndim != 3 or not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or x.shape[-1] != 4096 or x.numel() == 0
            or scale.shape != (x.shape[0], 1, 4096) or scale.dtype != torch.bfloat16
            or scale.device != x.device or not scale.is_contiguous()
            or norm.elementwise_affine or tuple(norm.normalized_shape) != (4096,)):
        return None
    rows, width = x.numel() // 4096, 4096
    padded = padded_rows(rows)
    q = torch.empty((padded, width), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(padded * width // 32, device=x.device, dtype=torch.uint8)
    _welford_scale_quantize[(padded,)](
        x, scale, q, scales, rows, x.shape[1], width // 128,
        D=width, EPS=norm.eps, SINGLE_BATCH=x.shape[0] == 1,
        num_warps=4, enable_fp_fusion=False)
    return Quantized(q, scales.view(torch.float8_e8m0fnu), rows, x.shape)


@triton.jit
def _swiglu_quantize(G, V, Q, S, M, K: tl.constexpr, KB_TILES):
    pid_m, pid_k = tl.program_id(0), tl.program_id(1)
    r, c = tl.arange(0, 128), tl.arange(0, 128)
    rows, cols = pid_m * 128 + r, pid_k * 128 + c
    index = rows[:, None].to(tl.int64) * K + cols[None, :]
    live = rows[:, None] < M
    g = tl.load(G + index, mask=live, other=0.0).to(tl.float32)
    v = tl.load(V + index, mask=live, other=0.0).to(tl.float32)
    # Same rounding as the BF16 SwiGLU: round SiLU to BF16, then an FP32
    # multiply that keeps subnormal operands a BF16 multiply would flush.
    silu = tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                                     args=[g, 1.0 + libdevice.exp(-g)], dtype=tl.float32,
                                     is_pure=True, pack=1).to(tl.bfloat16).to(tl.float32)
    product = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                                        args=[silu, v], dtype=tl.float32, is_pure=True, pack=1)
    # Same reason as the LayerNorm epilogue: match the BF16 store it replaces.
    blocks = tl.reshape(product.to(tl.bfloat16).to(tl.float32), (128, 4, 32))
    amax = tl.max(tl.abs(blocks), axis=2)
    bits = amax.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) - 8 + tl.where((bits & 0x7FFFFF) > 6291456, 1, 0)
    exponent = tl.where(amax == 0.0, 0, tl.maximum(tl.minimum(exponent, 254), 0))
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    tl.store(Q + index, tl.reshape(blocks * inverse[:, :, None], (128, 128)).to(Q.dtype.element_ty))
    offset = (r % 32)[:, None] * 16 + (r // 32)[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(S + (pid_m * KB_TILES + pid_k) * 512 + offset, exponent.to(tl.uint8))


def swiglu_quantize(gate, value):
    """SwiGLU and MXFP8 quantization in one pass, or None when unsupported."""
    if (gate.dtype != torch.bfloat16 or value.dtype != torch.bfloat16
            or not gate.is_cuda or gate.shape != value.shape
            or not gate.is_contiguous() or not value.is_contiguous()
            or gate.ndim != 2 or gate.shape[-1] % 128 or gate.numel() == 0):
        return None
    rows, width = gate.shape
    padded = padded_rows(rows)
    q = torch.empty((padded, width), device=gate.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(padded * width // 32, device=gate.device, dtype=torch.uint8)
    _swiglu_quantize[(padded // 128, width // 128)](
        gate, value, q, scales, rows, width, width // 128, num_warps=8, enable_fp_fusion=False)
    return Quantized(q, scales.view(torch.float8_e8m0fnu), rows, gate.shape)
