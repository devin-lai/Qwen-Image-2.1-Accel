# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""A same-shape request must replace graph prefix, RoPE, mask and timestep."""
def main():
    import torch
    from torch import nn

    from qwen_image_accel.optimizations.cuda_graph import install_cuda_graph
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    class Block(nn.Module):
        def forward(self, hidden_states, modulation, rotary_emb, attention_mask,
                    layer_cache, **kwargs):
            return (hidden_states + modulation[0, 0] + rotary_emb.real.mean()
                    + attention_mask.float().mean() + layer_cache.k.mean() + layer_cache.v.mean())

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([Block(), Block()])
            self._qw_modulation_cache = {}
        def forward(self, hidden_states, modulation, rotary_emb, attention_mask, kv_cache, kv_cache_mode):
            for i, block in enumerate(self.transformer_blocks):
                hidden_states = block(hidden_states=hidden_states, modulation=modulation,
                    rotary_emb=rotary_emb, attention_mask=attention_mask,
                    layer_cache=kv_cache.get_layer(i), kv_cache_mode=kv_cache_mode)
            return hidden_states

    transformer = Transformer()
    install_cuda_graph(transformer)
    with torch.no_grad():
        for request in [1, 2, 3]:
            cache = QwenImage21KVCache(2)
            for layer in cache.layer_caches:
                layer.store(torch.full((1, 5, 1, 1), float(request), device="cuda"),
                            torch.full((1, 5, 1, 1), float(request+1), device="cuda"))
            for timestep in [1, 2]:
                out = transformer(hidden_states=torch.ones(1, 3, 4, device="cuda"),
                    modulation=torch.full((2, 4), float(timestep), device="cuda"),
                    rotary_emb=torch.full((3, 2), complex(request), device="cuda"),
                    attention_mask=torch.full((1, 1, 1, 8), request % 2 == 1, device="cuda"),
                    kv_cache=cache, kv_cache_mode="cached")
                expected = 1 + 2 * (timestep + request + (request % 2) + request + request + 1)
                assert torch.equal(out, torch.full_like(out, expected)), (request, timestep, out, expected)
    assert len(transformer._qw_graphs) == 1
    print("PASS: changed same-shape requests and timesteps use fresh GPU values; graph count = 1")


if __name__ == "__main__":
    main()
