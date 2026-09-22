# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Choose the VAE decode tile grid from the sample size and the free VRAM.

The provided initialization calls ``vae.enable_tiling()`` with the checkpoint's
defaults: a 256-pixel tile advancing in 192-pixel strides. Those defaults are
sized for a GPU that cannot hold a whole decode, and on a 32 GiB RTX 5090 they
are the worst of both worlds. A 1024x1024 decode becomes 25 overlapping tiles
whose halos are recomputed, which costs 2.6x an untiled decode, and because the
overlaps are *blended* rather than exact the result also sits ~39 dB from the
decode it approximates. Tiling is buying memory that is already there.

This module keeps tiling where it is load-bearing and removes it where it is
not. Before each decode it estimates the untiled footprint from the sample area,
compares that with the free VRAM, and picks the coarsest grid that fits: one
tile -- no tiling at all -- whenever the decode fits, and otherwise the smallest
number of tiles that does, each still overlapping by ``BLEND`` pixels so the
blending the checkpoint relies on still happens.

The footprint is linear in output pixels and was measured on this GPU at
512x512 and 1024x1024 during the original stage experiments; ``BYTES_PER_PIXEL``
is that slope. An estimate can still be wrong -- another process may take memory
between the estimate and the allocation -- so an out-of-memory decode is caught
and retried on a finer grid rather than propagated.
"""
import collections
import math
import os

import torch

# Peak decode bytes per output pixel, BF16, measured at 512² (1.64 GiB) and
# 1024² (6.57 GiB) on this checkpoint: both give 6.72 KiB/pixel to three digits.
BYTES_PER_PIXEL = 6.72 * 1024

# Overlap kept between neighbouring tiles. The checkpoint's own default blend is
# 64 pixels (256 minus a 192 stride); 128 is a full extra step of the decoder's
# coarsest stage and costs one tile's worth of halo at the grids used here.
BLEND = 128

# Never plan a decode into more than this share of what is currently free.
BUDGET_FRACTION = 0.7

# How many recent grid choices to keep for inspection.
CHOICE_HISTORY = 32


def _plan(height, width, budget_bytes, alignment):
    """Smallest n×n tile grid whose single-tile footprint fits the budget."""
    for tiles in range(1, 9):
        tile_h = math.ceil(height / tiles / alignment) * alignment
        tile_w = math.ceil(width / tiles / alignment) * alignment
        if tiles == 1:
            if tile_h * tile_w * BYTES_PER_PIXEL <= budget_bytes:
                return 1, None, None
            continue
        blend = math.ceil(BLEND / alignment) * alignment
        if (tile_h + blend) * (tile_w + blend) * BYTES_PER_PIXEL <= budget_bytes:
            return tiles, (tile_h + blend, tile_w + blend), (tile_h, tile_w)
    return None, None, None


def install_adaptive_tiling(vae, budget_fraction=BUDGET_FRACTION):
    """Wrap ``vae.decode`` so each call picks its own tile grid.

    Returns the settings recorded in the pipeline configuration. The provided
    tile defaults are kept as the fallback for a sample too large to plan.
    """
    original = vae.decode
    ratio = vae.spatial_compression_ratio
    provided = (vae.tile_sample_min_height, vae.tile_sample_min_width,
                vae.tile_sample_stride_height, vae.tile_sample_stride_width)
    # Bounded: the service decodes once per request and runs indefinitely.
    chosen = collections.deque(maxlen=CHOICE_HISTORY)

    def apply(plan):
        tiles, tile, stride = plan
        if tiles == 1:
            vae.disable_tiling()
        elif tile is None:
            vae.enable_tiling(*provided)
        else:
            vae.use_tiling = True
            vae.tile_sample_min_height, vae.tile_sample_min_width = tile
            vae.tile_sample_stride_height, vae.tile_sample_stride_width = stride

    def decode(z, *args, **kwargs):
        height = z.shape[-2] * ratio
        width = z.shape[-1] * ratio
        free = torch.cuda.mem_get_info(z.device)[0]
        tiles, tile, stride = _plan(height, width, free * budget_fraction, ratio)
        if tiles is None:                      # too large to plan; keep the provided grid
            tiles, tile, stride = 0, None, None
        apply((tiles, tile, stride))
        chosen.append({"pixels": [height, width], "tiles": tiles, "tile": tile, "stride": stride})
        try:
            return original(z, *args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            # The estimate was optimistic, or something else took the memory.
            torch.cuda.empty_cache()
            apply((0, None, None))
            chosen[-1] = {"pixels": [height, width], "tiles": 0,
                      "fallback": "provided grid after OOM"}
            return original(z, *args, **kwargs)

    vae.decode = decode
    vae._qw_tile_choices = chosen
    return {"policy": "adaptive", "blend_pixels": BLEND,
            "bytes_per_pixel": BYTES_PER_PIXEL, "budget_fraction": budget_fraction,
            "provided_grid": {"tile": provided[:2], "stride": provided[2:]}}


def configure_vae(vae, approximate):
    """Apply ``QW21_VAE_TILING``; approximate profiles default to ``adaptive``.

    ``provided`` reproduces the original initialization exactly and is the
    default for the bit-exact profiles, because a different tile grid produces
    different pixels even though it is closer to an untiled decode.
    """
    choice = os.environ.get("QW21_VAE_TILING", "").strip() or ("adaptive" if approximate else "provided")
    if choice == "provided":
        vae.enable_tiling()
        return {"policy": "provided"}
    if choice == "off":
        vae.disable_tiling()
        return {"policy": "off"}
    if choice != "adaptive":
        raise ValueError(f"QW21_VAE_TILING must be provided, adaptive or off; got {choice!r}")
    vae.enable_tiling()
    return install_adaptive_tiling(vae)
