# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""SageAttention2 decode only; masked prefill uses native BF16 attention.

``QW21_SAGE_ACCUM`` selects the P.V accumulation. The default ``fp32+fp32`` is
the library's most accurate setting and the one every measurement in
docs/benchmarks/README.md used. ``fp16+fp32`` is far faster in isolation but
returned NaN at 16384 tokens on this GPU, so it is not a default; see
docs/benchmarks/README.md.
"""
import os

import torch

ACCUM = os.environ.get("QW21_SAGE_ACCUM", "fp32+fp32").strip()
QK_GRAN = os.environ.get("QW21_SAGE_QK_GRAN", "per_warp").strip()


@torch.library.custom_op("qw21::sage_decode", mutates_args=())
def sage_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    from sageattention import sageattn_qk_int8_pv_fp8_cuda
    return sageattn_qk_int8_pv_fp8_cuda(
        q, k, v, tensor_layout="NHD", is_causal=False,
        qk_quant_gran=QK_GRAN, pv_accum_dtype=ACCUM,
    )


@sage_decode.register_fake
def _(q, k, v):
    return torch.empty_like(q)
