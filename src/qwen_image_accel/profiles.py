# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Supported profiles, shared by the CLI, API and benchmark runners."""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Profile:
    description: str
    fused: bool = False
    quantized: bool = False
    sage: bool = False
    compile: bool = False
    graph: bool = False
    skip_blocks: str | None = None


PROFILES = MappingProxyType({
    "baseline": Profile("Original BF16 initialization"),
    "safe": Profile("Conservative encoder and modulation improvements"),
    "fused": Profile("Default: selective BF16 fusion", fused=True),
    "fused_sage": Profile("BF16 fusion with SageAttention2", fused=True, sage=True),
    "mxfp8": Profile("Block-scaled FP8 linears", fused=True, quantized=True),
    "mxfp8_sage": Profile("Block-scaled FP8 with SageAttention2", fused=True, quantized=True, sage=True),
    "mxfp8_balanced": Profile("MXFP8 with six BF16 boundary blocks", fused=True,
                              quantized=True, skip_blocks="0,1,2,29,30,31"),
    "mxfp8_balanced_sage": Profile("Balanced MXFP8 with SageAttention2", fused=True,
                                   quantized=True, sage=True, skip_blocks="0,1,2,29,30,31"),
    "compiled": Profile("Experimental bounded block compilation", compile=True),
    "graph": Profile("Experimental bounded CUDA Graph replay", fused=True, graph=True),
})

_REMOVED = {
    "fp8": "mxfp8", "fp8_sage": "mxfp8_sage",
    "fp8_sage_compiled": "mxfp8_sage", "sage": "fused_sage",
    "fused_cudnn": "fused with QW21_ATTENTION=_native_cudnn",
    "mxfp8_cudnn": "mxfp8 with QW21_ATTENTION=_native_cudnn",
}


def get_profile(name: str) -> Profile:
    """Resolve a profile before loading weights or changing a pipeline."""
    if name in _REMOVED:
        raise ValueError(f"Profile {name!r} was retired; use {_REMOVED[name]}. See docs/migration.md")
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown Qwen optimization profile {name!r}; choose {', '.join(PROFILES)}") from None
