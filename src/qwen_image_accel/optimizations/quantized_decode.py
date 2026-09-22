# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Compose the round-3 BF16 fusions with MXFP8 GEMMs on the cached-decode path.

The earlier quantized profiles bypassed every fused kernel, so they paid for
separate Q/K RMSNorm, rotary, prefix packing, LayerNorm, SwiGLU and residual
launches that the default profile had already removed. This module keeps those
kernels and changes only the four GEMM sites.

It also removes redundant activation quantization. ``to_q``/``to_k``/``to_v``
consume the same modulated hidden state, and the SwiGLU gate and value
projections consume the same normalized state, so each group quantizes once and
reuses the FP8 tensor and its scales. That saves three of the five activation
quantizations per block without changing any result.
"""
import types

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor

from qwen_image_accel.kernels.attention import prepare_attention_inputs
from qwen_image_accel.kernels.bf16 import swiglu
from qwen_image_accel.kernels.elementwise import residual, rope_qk
from qwen_image_accel.kernels.layernorm import layer_norm_scale
from qwen_image_accel.kernels.epilogue import layer_norm_scale_quantize, swiglu_quantize
from qwen_image_accel.optimizations.mxfp8 import MXFP8Linear
from qwen_image_accel.kernels.mxfp8 import Quantized, quantize


class MXFusedProcessor(QwenImage21AttnProcessor):
    """Cached-decode attention with shared Q/K/V activation quantization.

    ``sage`` routes the unmasked decode attention through SageAttention2. The
    fused kernel already emits the (batch, tokens, heads, dim) layout both
    backends expect, so this is a call-site swap and nothing else changes.
    """

    def __init__(self, sage=False):
        self.sage = sage

    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None,
                 layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
                 segments=None, key_valid=None):
        if kv_cache_mode != "cached":
            return super().__call__(attn, hidden_states, attention_mask, rotary_emb,
                                    layer_cache, kv_cache_mode, cache_write_slice,
                                    segments, key_valid)
        projections = (attn.to_q, attn.to_k, attn.to_v)
        if isinstance(hidden_states, Quantized):
            # The block's LayerNorm already emitted FP8; no BF16 tensor exists.
            lead = hidden_states.shape[:-1]
            q, k, v = (p.matmul(hidden_states).view(*lead, attn.heads, -1) for p in projections)
        elif all(isinstance(p, MXFP8Linear) for p in projections) and hidden_states.is_contiguous():
            shared = quantize(hidden_states.reshape(-1, hidden_states.shape[-1]))
            lead = hidden_states.shape[:-1]
            q, k, v = (p.matmul(shared).view(*lead, attn.heads, -1) for p in projections)
        else:
            q, k, v = (p(hidden_states).unflatten(-1, (attn.heads, -1)) for p in projections)
        cached_k, cached_v = layer_cache.get()
        prepared = prepare_attention_inputs(q, k, v, attn.norm_q, attn.norm_k,
                                            rotary_emb, cached_k, cached_v)
        if prepared is not None:
            q, k, v = prepared
        else:
            q, k = attn.norm_q(q).to(v.dtype), attn.norm_k(k).to(v.dtype)
            if rotary_emb is not None:
                q, k = rope_qk(q, k, rotary_emb)
            k, v = torch.cat([cached_k, k], dim=1), torch.cat([cached_v, v], dim=1)
        if self.sage and attention_mask is None and key_valid is None:
            # Imported here so profiles without Sage never register its custom op.
            from qwen_image_accel.optimizations.sage_attention import sage_decode
            result = sage_decode(q, k, v)
        else:
            result = dispatch_attention_fn(q, k, v, attn_mask=attention_mask, dropout_p=0.0,
                                           backend=self._attention_backend,
                                           parallel_config=self._parallel_config)
        return attn.to_out[1](attn.to_out[0](result.flatten(2, 3).to(q.dtype)))


def _mx_mlp_forward(self, hidden_states):
    """SwiGLU whose gate and value projections share one quantized activation."""
    lead = hidden_states.shape[:-1]
    if isinstance(hidden_states, Quantized):
        shared = hidden_states
    elif (isinstance(self.gate_layer, MXFP8Linear) and isinstance(self.proj, MXFP8Linear)
            and hidden_states.is_contiguous()):
        shared = quantize(hidden_states.reshape(-1, hidden_states.shape[-1]))
    else:
        return self.out(swiglu(self.gate_layer(hidden_states), self.proj(hidden_states)))
    gate, value = self.gate_layer.matmul(shared), self.proj.matmul(shared)
    if isinstance(self.out, MXFP8Linear):
        activated = swiglu_quantize(gate, value)
        if activated is not None:
            return self.out.matmul(activated).view(*lead, -1)
    return self.out(swiglu(gate, value)).view(*lead, -1)


def _normalized(hidden_states, scale, norm, *consumers):
    """FP8 straight out of the LayerNorm when every consumer is an MXFP8 linear."""
    if all(isinstance(consumer, MXFP8Linear) for consumer in consumers):
        fused = layer_norm_scale_quantize(hidden_states, scale, norm)
        if fused is not None:
            return fused
    return layer_norm_scale(hidden_states, scale, norm)


def install_mx_fused_decode(transformer, sage=False):
    """Install the composed decode path. Prefill keeps the native block forward."""
    cache = {"modulation": None, "values": None}
    transformer._qw_modulation_cache = cache
    for block in transformer.transformer_blocks:
        eager = block.forward
        block.attn.set_processor(MXFusedProcessor(sage=sage))
        block.img_mlp.forward = types.MethodType(_mx_mlp_forward, block.img_mlp)

        def forward(*args, _block=block, _eager=eager, **kwargs):
            if kwargs.get("kv_cache_mode") != "cached":
                return _eager(*args, **kwargs)
            h, modulation = kwargs["hidden_states"], kwargs["modulation"]
            if cache["modulation"] is not modulation:
                # The trailing row belongs to the cached prefix, not the target.
                s1, g1, s2, g2 = modulation[:-1].chunk(4, dim=-1)
                cache["modulation"] = modulation
                cache["values"] = (1 + s1[:, None], g1[:, None].tanh(),
                                   1 + s2[:, None], g2[:, None].tanh())
            s1, g1, s2, g2 = cache["values"]
            attn_kwargs = {k: v for k, v in kwargs.items()
                           if k not in {"hidden_states", "modulation", "target_token_mask"}}
            attn = _block.attn
            x = _normalized(h, s1, _block.img_norm1, attn.to_q, attn.to_k, attn.to_v)
            h = residual(h, attn(hidden_states=x, **attn_kwargs), g1)
            mlp = _block.img_mlp
            x = _normalized(h, s2, _block.img_norm2, mlp.gate_layer, mlp.proj)
            return residual(h, mlp(x), g2)

        block.forward = forward
