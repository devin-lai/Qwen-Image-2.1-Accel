# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Bounded CUDA Graph replay of target-only DiT blocks, with native prefill.

Each request copies its freshly built native BF16 prefix cache into graph-owned
GPU buffers. No prefix from another request is reused. The pipeline must be used
serially, as required by its mutable scheduler and existing offload hooks.
"""
from collections import OrderedDict
import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVLayerCache


def install_cuda_graph(transformer, max_graphs=1):
    forwards = [block.forward for block in transformer.transformer_blocks]
    context = {}
    graphs = OrderedDict()
    transformer._qw_graphs = graphs

    def pre_hook(module, args, kwargs):
        context["cache"] = kwargs.get("kv_cache")
    transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def first(*args, **kwargs):
        if kwargs.get("kv_cache_mode") != "cached":
            return forwards[0](*args, **kwargs)
        kv_cache = context["cache"]
        h, mod, rope = kwargs["hidden_states"], kwargs["modulation"], kwargs["rotary_emb"]
        mask = kwargs.get("attention_mask")
        key = (tuple(h.shape), tuple(mod.shape), tuple(rope.shape), str(h.dtype),
               tuple(kv_cache.get_layer(0).k.shape), None if mask is None else tuple(mask.shape))
        if key not in graphs:
            while len(graphs) >= max_graphs:
                graphs.popitem(last=False)
            static = {"h": h.clone(), "mod": mod.clone(), "rope": rope.clone(),
                      "mask": None if mask is None else mask.clone(), "caches": []}
            for layer in kv_cache.layer_caches:
                dst = QwenImage21KVLayerCache()
                dst.store(layer.k.clone(), layer.v.clone())
                static["caches"].append(dst)
            static_kwargs = dict(kwargs)
            static_kwargs.update(modulation=static["mod"], rotary_emb=static["rope"],
                                 attention_mask=static["mask"])
            def body():
                # Record shared modulation arithmetic inside the graph, so each
                # replay reads the new timestep instead of old gate values.
                transformer._qw_modulation_cache["modulation"] = None
                value = static["h"]
                for i, forward in enumerate(forwards):
                    call = {**static_kwargs, "hidden_states": value,
                            "layer_cache": static["caches"][i]}
                    value = forward(**call)
                return value
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    body()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = body()
            graphs[key] = (graph, static, output, kv_cache)
        graph, static, output, last_cache = graphs[key]
        graphs.move_to_end(key)
        static["h"].copy_(h)
        static["mod"].copy_(mod)
        # RoPE/mask and prefix are request-specific even when tensor shapes match.
        if last_cache is not kv_cache:
            static["rope"].copy_(rope)
            if mask is not None:
                static["mask"].copy_(mask)
            for dst, src in zip(static["caches"], kv_cache.layer_caches):
                dst.k.copy_(src.k)
                dst.v.copy_(src.v)
            graphs[key] = (graph, static, output, kv_cache)
        graph.replay()
        return output

    transformer.transformer_blocks[0].forward = first
    for index, block in enumerate(transformer.transformer_blocks[1:], 1):
        def forward(*args, _eager=forwards[index], **kwargs):
            if kwargs.get("kv_cache_mode") == "cached":
                return kwargs["hidden_states"]
            return _eager(*args, **kwargs)
        block.forward = forward
