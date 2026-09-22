# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Opt-in Qwen-Image-2.1 optimizations, scoped to one pipeline instance.

The native causal prefill and BF16 prefix KV cache are retained in every profile.
Approximate arithmetic is never enabled implicitly.
"""
import types
import hashlib
import os
from importlib.metadata import version
from pathlib import Path

import torch
from diffusers.hooks import apply_group_offloading

from .profiles import get_profile


def _select_attention_backend(transformer):
    """Return the decode attention backend, honouring QW21_ATTENTION when set."""
    override = os.environ.get("QW21_ATTENTION", "").strip()
    if override:
        backend = None if override in {"native", "default"} else override
    else:
        return "native"
    for block in transformer.transformer_blocks:
        block.attn.processor._attention_backend = backend
    return backend or "native"


def _place_text_encoder(module):
    """Stream the encoder's weights from CPU, or keep it resident when asked.

    The provided initialization streams the 16.33 GiB BF16 encoder one leaf at a
    time, on every request. That costs a fixed ~0.305 s, which the BF16 profiles
    hide behind a much longer denoise but the quantized profiles do not: at
    512x512 it is a sixth of the whole request. Residency removes it and is
    exactly the same arithmetic, but it needs the 16.33 GiB, which only leaves
    room beside a quantized DiT. It stays opt-in because running out of VRAM
    mid-request is a worse failure than 0.305 s.

    ``QW21_TEXT_ENCODER`` takes ``leaf`` (the provided behaviour, default),
    ``resident``, or ``block_<n>`` for coarser streaming groups.
    """
    choice = os.environ.get("QW21_TEXT_ENCODER", "leaf").strip()
    if choice == "resident":
        module.to("cuda")
        return "resident"
    kwargs = {"offload_type": "leaf_level"}
    if choice.startswith("block_"):
        kwargs = {"offload_type": "block_level", "num_blocks_per_group": int(choice[6:])}
    elif choice not in {"leaf", "offload"}:
        raise ValueError(f"QW21_TEXT_ENCODER must be leaf, resident or block_<n>; got {choice!r}")
    apply_group_offloading(module, onload_device=torch.device("cuda"),
                           offload_device=torch.device("cpu"),
                           use_stream=True, record_stream=False, **kwargs)
    return kwargs.get("offload_type") if "num_blocks_per_group" not in kwargs else choice


def _encoder_features_only(self, *args, **kwargs):
    # Pipeline consumes hidden_states, never vocabulary logits. Keep its pre-norm
    # hook intact and avoid allocating a text-generation KV cache.
    kwargs["use_cache"] = False
    return self.model(*args, **kwargs)


def _optimize_decode_blocks(transformer, compile_blocks=False):
    # Preserve explicit BF16 rounding at eager operation boundaries. Disabling
    # this can alter iterative diffusion output even without quantized weights.
    if compile_blocks:
        import torch._inductor.config
        torch._inductor.config.emulate_precision_casts = True
        from qwen_image_accel.optimizations.compile import BoundedCompile
        max_shapes = int(os.environ.get("QW21_COMPILE_MAX_SHAPES", "2"))
    for block in transformer.transformer_blocks:
        eager = block.forward
        compiled = BoundedCompile(eager, max_shapes=max_shapes) if compile_blocks else eager
        if compile_blocks:
            block._qw_compile_guard = compiled
        def forward(*args, _eager=eager, _compiled=compiled, **kwargs):
            if kwargs.get("kv_cache_mode") == "cached":
                # Native cached decode contains ONLY target-image rows. Broadcast
                # each sample's timestep modulation instead of materializing four
                # [B, image_tokens, hidden_dim] where() results in every block.
                # The trailing t=0 row belongs to the cached prefix, not the target.
                kwargs["modulation"] = kwargs["modulation"][:-1]
                kwargs["target_token_mask"] = None
                return _compiled(*args, **kwargs)
            return _eager(*args, **kwargs)
        block.forward = forward


def _validate_fused_runtime(pipe):
    """Reject incompatible inputs before installing any GPU patch."""
    if pipe.transformer.dtype != torch.bfloat16:
        raise ValueError("The validated fused/graph profiles require a BF16 transformer")
    if (torch.__version__.split("+")[0] != "2.11.0" or torch.version.cuda != "13.0"
            or version("triton") != "3.6.0" or torch.cuda.get_device_capability() != (12, 0)):
        raise RuntimeError("Fused kernels were validated on torch 2.11.0+cu130 / Triton 3.6.0 / SM120; "
                           "use QW21_PROFILE=safe on an unvalidated runtime")
    from diffusers.models.transformers import transformer_qwenimage21 as native
    expected = "0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7"
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Qwen transformer source differs from the validated bundled Diffusers wheel; "
                           "run scripts/install.sh or use QW21_PROFILE=safe")


def configure_pipeline(pipe, profile="fused"):
    selected = get_profile(profile)
    if hasattr(pipe, "_qw_optimization"):
        raise RuntimeError("Configure a fresh pipeline once; profiles cannot be switched in place")
    if selected.fused:
        _validate_fused_runtime(pipe)
    if profile == "baseline":
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        apply_group_offloading(pipe.text_encoder, onload_device=torch.device("cuda"),
                               offload_device=torch.device("cpu"), offload_type="leaf_level",
                               use_stream=True, record_stream=False)
        pipe.vae.enable_tiling()
        pipe._qw_optimization = {"profile": "baseline", "approximate": False}
        return pipe._qw_optimization
    pipe.text_encoder.forward = types.MethodType(_encoder_features_only, pipe.text_encoder)
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    placement = _place_text_encoder(pipe.text_encoder.model)
    config = {"profile": profile, "features_only_encoder": True,
              "native_bf16_prefix_cache": True, "dit_gpu_resident": True,
              "text_encoder_offload": placement, "approximate": False}
    if selected.quantized:
        from qwen_image_accel.optimizations.mxfp8 import quantize_transformer as quantize_mxfp8
        skip = os.environ.get("QW21_MXFP8_SKIP_BLOCKS", selected.skip_blocks or "0,31")
        layers = os.environ.get("QW21_MXFP8_LAYERS", "all")
        config["mxfp8_linears"] = len(quantize_mxfp8(pipe.transformer, skip, layers))
        config["mxfp8_skipped_blocks"] = skip
        config["mxfp8_layer_selection"] = layers
        config["approximate"] = True
    if selected.sage:
        try:
            from sageattention import sageattn_qk_int8_pv_fp8_cuda
        except ImportError as exc:
            raise RuntimeError("Install the optional SageAttention2 dependency with scripts/install_sageattention.sh; "
                               "see docs/optimization.md") from exc
        from .optimizations import sage_attention
        config["sage_decode"] = True
        config["sage_pv_accum"] = sage_attention.ACCUM
        config["sage_qk_gran"] = sage_attention.QK_GRAN
        config["approximate"] = True
    if selected.fused:
        composed = selected.quantized or selected.sage
        if composed:
            # One composed decode path: fused kernels, optional MXFP8 GEMMs and
            # optional SageAttention2. On a BF16 model with sage disabled it is
            # equivalent to `install_fused_decode`.
            from qwen_image_accel.optimizations.quantized_decode import install_mx_fused_decode
            install_mx_fused_decode(pipe.transformer, sage=selected.sage)
        else:
            from qwen_image_accel.optimizations.fusion import install_fused_decode
            install_fused_decode(pipe.transformer)
        from qwen_image_accel.kernels.bf16 import install_qk_norm, install_swiglu
        install_qk_norm(pipe.transformer)
        if not composed:
            install_swiglu(pipe.transformer)   # the composed path installs its own
        config["fused_qk_rmsnorm"] = True
        config["fused_swiglu"] = True
        config["fused_norm_rope_kv_pack"] = True
        config["fused_native_order_layernorm_scale"] = True
        config["runtime_token_counts"] = True
        if selected.graph:
            from qwen_image_accel.optimizations.cuda_graph import install_cuda_graph
            install_cuda_graph(pipe.transformer)
    else:
        _optimize_decode_blocks(pipe.transformer, compile_blocks=selected.compile)
    config["attention_backend"] = _select_attention_backend(pipe.transformer)
    if selected.compile or config["attention_backend"] != "native":
        config["approximate"] = True
    config["broadcast_decode_modulation"] = True
    config["compiled_decode_blocks"] = selected.compile
    if selected.compile:
        config["compile_max_shapes_per_block"] = int(os.environ.get("QW21_COMPILE_MAX_SHAPES", "2"))
        config["compile_unseen_shape_policy"] = "eager after shape budget"
    config["cuda_graph"] = selected.graph
    # Orthogonal to the profile: reuses whole-network residuals across steps, so
    # it must wrap whatever block forward the profile installed.
    from qwen_image_accel.optimizations.step_cache import configure_from_environment
    step_cache = configure_from_environment(pipe.transformer)
    if step_cache is not None:
        config["step_cache"] = step_cache
        config["approximate"] = True
    # Preserve the measured tile defaults; compilation and attention overrides
    # alone have historically kept the provided grid.
    from qwen_image_accel.optimizations.vae import configure_vae
    config["vae_tiling"] = configure_vae(pipe.vae, selected.quantized or selected.sage or step_cache is not None)
    if config["vae_tiling"]["policy"] != "provided":
        config["approximate"] = True
    pipe._qw_optimization = config
    return config
