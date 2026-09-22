# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""BF16 pointwise fusion with explicit rounding; no whole-block compilation."""
import types
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _swiglu_kernel(G, X, O, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + i, i < N, 0).to(tl.float32)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    # Match eager SiLU rounding before multiplying the other BF16 projection.
    s = tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                                  args=[g, 1.0 + libdevice.exp(-g)], dtype=tl.float32,
                                  is_pure=True, pack=1).to(tl.bfloat16).to(tl.float32)
    # Prevent lowering this to BF16 mul, which flushes subnormal operands.
    product = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                                       args=[s, x], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(O + i, product, i < N)


def swiglu(gate, value):
    if (gate.dtype != torch.bfloat16 or value.dtype != torch.bfloat16
            or not gate.is_cuda or not value.is_cuda
            or not gate.is_contiguous() or not value.is_contiguous()
            or gate.shape != value.shape):
        return F.silu(gate) * value
    out = torch.empty_like(gate)
    _swiglu_kernel[(triton.cdiv(gate.numel(), 1024),)](
        gate, value, out, gate.numel(), BLOCK=1024, enable_fp_fusion=False)
    return out


def install_swiglu(transformer):
    for block in transformer.transformer_blocks:
        def forward(self, hidden_states):
            return self.out(swiglu(self.gate_layer(hidden_states), self.proj(hidden_states)))
        block.img_mlp.forward = types.MethodType(forward, block.img_mlp)


@triton.jit
def _rms_kernel(X, W, O, ROWS, D: tl.constexpr, EPS: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, D)
    x = tl.load(X + row[:, None] * D + col[None, :], row[:, None] < ROWS, 0).to(tl.float32)
    weight = tl.load(W + col).to(tl.float32)
    variance = tl.sum(x * x, axis=1) / D
    # Diffusers RMSNorm rounds to BF16 BEFORE multiplying its BF16 weight.
    normalized = (x * tl.rsqrt(variance[:, None] + EPS)).to(tl.bfloat16).to(tl.float32)
    tl.store(O + row[:, None] * D + col[None, :], normalized * weight[None, :], row[:, None] < ROWS)


def rms_norm(x, weight, eps):
    out = torch.empty_like(x)
    rows = x.numel() // x.shape[-1]
    _rms_kernel[(triton.cdiv(rows, 4),)](
        x, weight, out, rows, D=x.shape[-1], EPS=eps, R=4, enable_fp_fusion=False)
    return out


def install_qk_norm(transformer):
    for block in transformer.transformer_blocks:
        for norm in (block.attn.norm_q, block.attn.norm_k):
            eager = norm.forward
            def forward(self, x, _eager=eager):
                if (x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
                        and x.shape[-1] == 128 and self.weight is not None
                        and self.weight.dtype == torch.bfloat16 and self.weight.is_contiguous()
                        and self.bias is None):
                    return rms_norm(x, self.weight, self.eps)
                return _eager(x)
            norm.forward = types.MethodType(forward, norm)
