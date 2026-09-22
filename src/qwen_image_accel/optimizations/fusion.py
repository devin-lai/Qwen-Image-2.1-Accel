# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Selective BF16 fusion. Native GEMMs, attention and prefill are retained."""
import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor
from qwen_image_accel.kernels.elementwise import residual, rope_qk
from qwen_image_accel.kernels.attention import prepare_attention_inputs
from qwen_image_accel.kernels.layernorm import layer_norm_scale


class FusedRopeProcessor(QwenImage21AttnProcessor):
    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None,
                 layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
                 segments=None, key_valid=None):
        if kv_cache_mode != "cached":
            return super().__call__(attn, hidden_states, attention_mask, rotary_emb,
                                    layer_cache, kv_cache_mode, cache_write_slice, segments, key_valid)
        q = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        k = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
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
        result = dispatch_attention_fn(q, k, v, attn_mask=attention_mask, dropout_p=0.0,
                                       backend=self._attention_backend,
                                       parallel_config=self._parallel_config)
        return attn.to_out[1](attn.to_out[0](result.flatten(2, 3).to(q.dtype)))


def install_fused_decode(transformer):
    cache = {"modulation": None, "values": None}
    transformer._qw_modulation_cache = cache
    for block in transformer.transformer_blocks:
        eager = block.forward
        block.attn.set_processor(FusedRopeProcessor())
        def forward(*args, _block=block, _eager=eager, **kwargs):
            if kwargs.get("kv_cache_mode") != "cached":
                return _eager(*args, **kwargs)
            h, modulation = kwargs["hidden_states"], kwargs["modulation"]
            if cache["modulation"] is not modulation:
                s1, g1, s2, g2 = modulation[:-1].chunk(4, dim=-1)
                cache["modulation"] = modulation
                cache["values"] = (1 + s1[:, None], g1[:, None].tanh(),
                                   1 + s2[:, None], g2[:, None].tanh())
            s1, g1, s2, g2 = cache["values"]
            x = layer_norm_scale(h, s1, _block.img_norm1)
            attn_kwargs = {k: v for k, v in kwargs.items()
                           if k not in {"hidden_states", "modulation", "target_token_mask"}}
            h = residual(h, _block.attn(hidden_states=x, **attn_kwargs), g1)
            x = layer_norm_scale(h, s2, _block.img_norm2)
            return residual(h, _block.img_mlp(x), g2)
        block.forward = forward
