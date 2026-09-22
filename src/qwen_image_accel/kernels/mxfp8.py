# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Block-scaled E4M3 quantization and the NVIDIA scale layout."""
from typing import NamedTuple

import torch
import triton
import triton.language as tl


class Quantized(NamedTuple):
    """An MXFP8 activation: padded E4M3 values plus their swizzled E8M0 scales."""

    values: torch.Tensor     # (rows padded to 128, K) float8_e4m3fn
    scales: torch.Tensor     # flat float8_e8m0fnu in NVIDIA's 128x4 block order
    rows: int                # live rows, before padding
    shape: torch.Size        # logical shape of the tensor this came from


def padded_rows(rows):
    """MXFP8 scales are tiled over 128 rows, so M must be a multiple of 128."""
    return (rows + 127) // 128 * 128


BLOCK = tl.constexpr(32)        # MX block length along K
ROW_TILE = tl.constexpr(128)    # rows per scale tile
K_TILE = tl.constexpr(128)      # = 4 scale blocks, one 512-entry swizzle tile


@triton.jit
def _mx_quantize_kernel(X, Q, S, M, K: tl.constexpr, stride_xm, KB_TILES):
    """Quantize one 128x128 tile to E4M3 and emit its 512 swizzled E8M0 scales."""
    pid_m, pid_k = tl.program_id(0), tl.program_id(1)
    r = tl.arange(0, ROW_TILE)
    c = tl.arange(0, K_TILE)
    rows = pid_m * ROW_TILE + r
    cols = pid_k * K_TILE + c
    x = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + cols[None, :],
                mask=rows[:, None] < M, other=0.0).to(tl.float32)
    blocks = tl.reshape(x, (ROW_TILE, K_TILE // BLOCK, BLOCK))
    amax = tl.max(tl.abs(blocks), axis=2)
    # Smallest power of two with amax / 2**e <= 448. For amax = m * 2**ea with
    # m in [1, 2), that is ea - 8 unless m exceeds 448 / 256 = 1.75, in which
    # case one more binade is needed. Reading the exponent and mantissa from the
    # FP32 bits avoids a log2 whose rounding could clip the block maximum.
    bits = amax.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) - 8 + tl.where((bits & 0x7FFFFF) > 6291456, 1, 0)
    exponent = tl.where(amax == 0.0, 0, tl.maximum(tl.minimum(exponent, 254), 0))
    # 2**(127 - e) as an exact FP32 bit pattern; multiplying avoids a divide.
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    q = tl.reshape(blocks * inverse[:, :, None], (ROW_TILE, K_TILE))
    tl.store(Q + rows[:, None].to(tl.int64) * K + cols[None, :], q.to(Q.dtype.element_ty))
    # NVIDIA 128x4 blocked scale layout: within a (128 rows x 4 blocks) tile the
    # 512 entries are ordered (row % 32, row // 32, block).
    offset = (r % 32)[:, None] * 16 + (r // 32)[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(S + (pid_m * KB_TILES + pid_k) * 512 + offset, exponent.to(tl.uint8))


def quantize(x):
    """Quantize a contiguous BF16 ``(M, K)`` activation to MXFP8."""
    m, k = x.shape
    padded = padded_rows(m)
    q = torch.empty((padded, k), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(padded * k // 32, device=x.device, dtype=torch.uint8)
    _mx_quantize_kernel[(padded // 128, k // 128)](
        x, q, scale, m, k, x.stride(0), k // 128, num_warps=8)
    return Quantized(q, scale.view(torch.float8_e8m0fnu), m, x.shape)


