# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Replace selected transformer linears with Blackwell block-scaled MXFP8.

The quantizer and padded scale layout live in kernels/mxfp8.py. This module
owns module conversion, BF16 block exclusions and shared-activation GEMMs.
"""
import os

import torch
from torch import nn
from qwen_image_accel.kernels.mxfp8 import quantize


def _supported(k, n):
    return k % 128 == 0 and n % 128 == 0


class MXFP8Linear(nn.Module):
    """Bias-free linear whose weight is stored as MXFP8 and multiplied as MXFP8."""

    def __init__(self, linear):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("Qwen 2.1 DiT MXFP8 conversion expects bias-free linears")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        weight = linear.weight.detach().to(torch.bfloat16).contiguous()
        quantized = quantize(weight)
        self.register_buffer("weight_fp8", quantized.values.t())   # (K, N) column-major
        self.register_buffer("weight_scale", quantized.scales)

    def matmul(self, activation, dtype=torch.bfloat16):
        """Multiply an already quantized activation; several weights can share one."""
        y = torch._scaled_mm(activation.values, self.weight_fp8, scale_a=activation.scales,
                             scale_b=self.weight_scale, out_dtype=dtype)
        return y[:activation.rows]

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        activation = quantize(flat if flat.is_contiguous() else flat.contiguous())
        return self.matmul(activation, x.dtype).reshape(*x.shape[:-1], self.out_features)


# Which of a block's linears each selection converts. `attention` covers the
# four projections around attention, `mlp` the three SwiGLU projections.
LAYER_GROUPS = {
    "attention": ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"),
    "mlp": ("img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"),
}
LAYER_GROUPS["all"] = LAYER_GROUPS["attention"] + LAYER_GROUPS["mlp"]


def selected_layers(selection=None):
    """Resolve QW21_MXFP8_LAYERS: a group name or an explicit comma-separated list."""
    raw = (selection if selection is not None else os.environ.get("QW21_MXFP8_LAYERS", "all")).strip()
    if raw in LAYER_GROUPS:
        return set(LAYER_GROUPS[raw])
    names = {n.strip() for n in raw.split(",") if n.strip()}
    unknown = names - set(LAYER_GROUPS["all"])
    if unknown:
        raise ValueError(f"Unknown MXFP8 layer names {sorted(unknown)}; "
                         f"choose from {sorted(LAYER_GROUPS['all'])} or {sorted(LAYER_GROUPS)}")
    return names


def quantize_transformer(transformer, skip_blocks=None, layers=None):
    """Replace selected DiT block linears with MXFP8; everything else stays BF16.

    Norms, modulation, embeddings, the output projection, the VAE, the text
    encoder and the KV cache are never touched. Blocks listed in
    ``QW21_MXFP8_SKIP_BLOCKS`` keep every linear in BF16, which is how the
    higher-accuracy profiles buy back quality at the ends of the stack.
    """
    raw = skip_blocks if skip_blocks is not None else os.environ.get("QW21_MXFP8_SKIP_BLOCKS", "0,31")
    skip = {int(i) for i in str(raw).split(",") if i.strip()}
    wanted = selected_layers(layers)
    converted = []
    with torch.no_grad():
        for index, block in enumerate(transformer.transformer_blocks):
            if index in skip:
                continue
            for name in sorted(wanted):
                child = block.get_submodule(name)
                if not isinstance(child, nn.Linear) or not _supported(child.in_features, child.out_features):
                    continue
                parent_name, _, leaf = name.rpartition(".")
                parent = block.get_submodule(parent_name) if parent_name else block
                setattr(parent, leaf, MXFP8Linear(child))
                converted.append(f"transformer_blocks.{index}.{name}")
    torch.cuda.empty_cache()
    return converted
