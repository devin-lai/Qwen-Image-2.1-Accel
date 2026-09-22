# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Qwen-Image-2.1 inference with explicit, pipeline-scoped optimizations.

Importing this package does not load PyTorch, Triton or model weights.
"""
from .api import Init, Process
from .profiles import PROFILES

__version__ = "0.1.0"
__all__ = ["Init", "Process", "PROFILES", "configure_pipeline"]


def configure_pipeline(pipe, profile="fused"):
    """Configure an existing BF16 Diffusers pipeline once, before inference."""
    from .pipeline import configure_pipeline as configure
    return configure(pipe, profile)
