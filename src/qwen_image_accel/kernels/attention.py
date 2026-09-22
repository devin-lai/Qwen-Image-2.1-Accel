# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Fuse cached-decode Q/K RMSNorm, RoPE and prefix K/V packing.

Preserves both BF16 norm rounding points and native CUDA complex-FMA order.
Token and prefix lengths are runtime arguments. No persistent request buffers.
The caller retains the native path for unsupported inputs and for prefill.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _norm_rope_pack(
    Q,
    K,
    V,
    CK,
    CV,
    WQ,
    WK,
    F,
    OQ,
    OK,
    OV,
    ROWS,
    SEQ,
    PREFIX,
    H: tl.constexpr,
    D: tl.constexpr,
    EPSQ: tl.constexpr,
    EPSK: tl.constexpr,
    R: tl.constexpr,
    SINGLE_BATCH: tl.constexpr,
):
    if tl.program_id(0) < tl.cdiv(ROWS, R):
        row = tl.program_id(0) * R + tl.arange(0, R)
        col = tl.arange(0, D)
        index = row[:, None] * D + col[None, :]
        mask = row[:, None] < ROWS
        q = tl.load(Q + index, mask, 0).to(tl.float32)
        k = tl.load(K + index, mask, 0).to(tl.float32)
        wq = tl.load(WQ + col).to(tl.float32)
        wk = tl.load(WK + col).to(tl.float32)
        # Native RMSNorm rounds once before weight multiplication, then again
        # before complex RoPE. Removing either boundary changes diffusion output.
        q = (
            (q * tl.rsqrt(tl.sum(q * q, 1)[:, None] / D + EPSQ))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        k = (
            (k * tl.rsqrt(tl.sum(k * k, 1)[:, None] / D + EPSK))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        q = (q * wq[None, :]).to(tl.bfloat16).to(tl.float32)
        k = (k * wk[None, :]).to(tl.bfloat16).to(tl.float32)
        other = tl.broadcast_to((col ^ 1)[None, :], (R, D))
        pq = tl.gather(q, other, axis=1)
        pk = tl.gather(k, other, axis=1)
        token = row // H
        if not SINGLE_BATCH:
            token = token % SEQ
        fi = token[:, None] * D + (col[None, :] // 2) * 2
        c = tl.load(F + fi, mask, 0)
        s = tl.load(F + fi + 1, mask, 0)
        sq = tl.where(col[None, :] % 2 == 0, -(pq * s), pq * s)
        sk = tl.where(col[None, :] % 2 == 0, -(pk * s), pk * s)
        tl.store(OQ + index, libdevice.fma(q, c, sq), mask)
        packed_index = index + (row[:, None] // (SEQ * H) + 1) * PREFIX * H * D
        tl.store(OK + packed_index, libdevice.fma(k, c, sk), mask)
        v = tl.load(V + index, mask, 0)
        tl.store(OV + packed_index, v, mask)
    else:
        # Separate programs copy this call's prefix; no prefix survives a call.
        row = (tl.program_id(0) - tl.cdiv(ROWS, R)) * R + tl.arange(0, R)
        col = tl.arange(0, D)
        prefix_rows = (ROWS // SEQ) * PREFIX
        index = row[:, None] * D + col[None, :]
        mask = row[:, None] < prefix_rows
        dest = index + (row[:, None] // (PREFIX * H)) * SEQ * H * D
        prefix_k = tl.load(CK + index, mask, 0)
        prefix_v = tl.load(CV + index, mask, 0)
        tl.store(OK + dest, prefix_k, mask)
        tl.store(OV + dest, prefix_v, mask)


def prepare_attention_inputs(q, k, v, norm_q, norm_k, freqs, cached_k, cached_v):
    """Return fused Q/K/V, or None when the original operators should be used."""
    if (
        q.ndim != 4
        or q.shape[-2:] != (32, 128)
        or q.shape[1] == 0
        or k.shape != q.shape
        or v.shape != q.shape
        or freqs is None
        or freqs.dtype != torch.complex64
        or freqs.shape != (q.shape[1], 64)
        or not freqs.is_contiguous()
        or freqs.device != q.device
        or cached_k.ndim != 4
        or cached_k.shape[0] != q.shape[0]
        or cached_k.shape[2:] != q.shape[2:]
        or cached_v.shape != cached_k.shape
    ):
        return None
    tensors = (q, k, v, cached_k, cached_v)
    if any(
        t.dtype != torch.bfloat16
        or not t.is_cuda
        or t.device != q.device
        or not t.is_contiguous()
        for t in tensors
    ):
        return None
    for norm in (norm_q, norm_k):
        if (
            norm.weight is None
            or norm.bias is not None
            or norm.weight.shape != (128,)
            or norm.weight.dtype != torch.bfloat16
            or norm.weight.device != q.device
            or not norm.weight.is_contiguous()
        ):
            return None
    b, s, h, d = q.shape
    p = cached_k.shape[1]
    oq = torch.empty_like(q)
    ok = torch.empty((b, s + p, h, d), device=q.device, dtype=q.dtype)
    ov = torch.empty_like(ok)
    rows = b * s * h
    _norm_rope_pack[(triton.cdiv(rows, 4) + triton.cdiv(b * p * h, 4),)](
        q,
        k,
        v,
        cached_k,
        cached_v,
        norm_q.weight,
        norm_k.weight,
        torch.view_as_real(freqs),
        oq,
        ok,
        ov,
        rows,
        s,
        p,
        H=h,
        D=d,
        EPSQ=norm_q.eps,
        EPSK=norm_k.eps,
        R=4,
        SINGLE_BATCH=b == 1,
        enable_fp_fusion=False,
    )
    return oq, ok, ov
