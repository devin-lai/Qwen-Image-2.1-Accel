# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""BF16 gated residual and rotary embedding kernels."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _residual_kernel(H, X, G, O, N, D: tl.constexpr, S,
                     SINGLE_BATCH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    h = tl.load(H + i, i < N, 0).to(tl.float32)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    if SINGLE_BATCH:
        g_index = i % D
    else:
        g_index = (i // (S * D)) * D + i % D
    g = tl.load(G + g_index, i < N, 0).to(tl.float32)
    # Eager BF16 rounds multiplication before addition; keep that boundary.
    product = (x * g).to(tl.bfloat16).to(tl.float32)
    tl.store(O + i, h + product, i < N)


def residual(h, x, gate):
    if h.dtype != torch.bfloat16 or not h.is_contiguous() or not x.is_contiguous():
        return h + gate * x
    out = torch.empty_like(h)
    _residual_kernel[(triton.cdiv(h.numel(), 1024),)](
        h, x, gate, out, h.numel(), h.shape[-1], h.shape[-2],
        SINGLE_BATCH=h.shape[0] == 1, BLOCK=1024,
        enable_fp_fusion=False)
    return out


@triton.jit
def _rope_kernel(Q, K, F, OQ, OK, N, H: tl.constexpr,
                 D: tl.constexpr, S, SINGLE_BATCH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair = i * 2
    # Each complex pair uses one frequency, shared across heads and batch.
    token = i // (H * (D // 2))
    if not SINGLE_BATCH:
        token = token % S
    fi = (token * (D // 2) + i % (D // 2)) * 2
    c = tl.load(F + fi, i < N, 0)
    s = tl.load(F + fi + 1, i < N, 0)
    qr = tl.load(Q + pair, i < N, 0).to(tl.float32)
    qi = tl.load(Q + pair + 1, i < N, 0).to(tl.float32)
    kr = tl.load(K + pair, i < N, 0).to(tl.float32)
    ki = tl.load(K + pair + 1, i < N, 0).to(tl.float32)
    # Match the installed CUDA complex multiply's FMA operand order. A plain
    # algebraically equivalent expression differs at BF16 rounding boundaries.
    orq, ork = libdevice.fma(qr, c, -(qi * s)), libdevice.fma(kr, c, -(ki * s))
    oiq, oik = libdevice.fma(qi, c, qr * s), libdevice.fma(ki, c, kr * s)
    tl.store(OQ + pair, orq, i < N)
    tl.store(OQ + pair + 1, oiq, i < N)
    tl.store(OK + pair, ork, i < N)
    tl.store(OK + pair + 1, oik, i < N)


def rope_qk(q, k, freqs):
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    n = q.numel() // 2
    _rope_kernel[(triton.cdiv(n, 512),)](
        q, k, torch.view_as_real(freqs), out_q, out_k, n, q.shape[-2], q.shape[-1],
        q.shape[1], SINGLE_BATCH=q.shape[0] == 1, BLOCK=512, enable_fp_fusion=False)
    return out_q, out_k


