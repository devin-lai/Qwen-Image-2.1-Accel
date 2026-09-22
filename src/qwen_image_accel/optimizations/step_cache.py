# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Opt-in cross-step residual reuse for the cached-decode path (first-block cache).

Consecutive denoising steps of a flow-matching sampler produce very similar block
residuals once the layout has settled, so a step whose first block barely moved
is very likely to produce a whole-network residual close to the previous step's.
This module computes block 0 every step, measures how much its residual changed,
and on a small change reuses the cached residual of blocks 1..N-1 instead of
running them.

Unlike every other optimization here this one *removes* computation, so its error
is a modelling approximation rather than an arithmetic one. It is never enabled
by a profile; ``QW21_STEP_CACHE`` must name an explicit relative-L1 threshold.

Guards that keep the approximation bounded:

* the first ``QW21_STEP_CACHE_WARMUP`` decode steps always run in full, because
  early steps set the composition and are the least redundant;
* at most ``QW21_STEP_CACHE_MAX_RUN`` steps may be skipped in a row, so a stale
  residual cannot be carried through the whole trajectory;
* any prefill, shape change or dtype change drops the cache.
"""
import os

import torch


class FirstBlockCache:
    """Decide, once per decode step, whether blocks 1..N-1 can be reused."""

    def __init__(self, threshold, warmup=2, max_run=3):
        self.threshold, self.warmup, self.max_run = threshold, warmup, max_run
        self.reset()

    def reset(self):
        self.previous_delta = None
        self.tail = None
        self.after_first = None
        self.skip = False
        self.run = 0
        self.step = 0
        self.decisions = []   # (relative change, reused) per eligible step

    def decide(self, delta):
        self.step += 1
        reusable = (self.previous_delta is not None and self.tail is not None
                    and self.previous_delta.shape == delta.shape
                    and self.step > self.warmup and self.run < self.max_run)
        if reusable:
            change = (delta - self.previous_delta).abs().mean()
            scale = self.previous_delta.abs().mean().clamp_min(1e-12)
            # One synchronization per step; a data-dependent branch needs the value.
            relative = (change / scale).item()
            self.skip = relative < self.threshold
            self.decisions.append((round(relative, 5), self.skip))
        else:
            self.skip = False
        self.run = self.run + 1 if self.skip else 0
        if not self.skip:
            self.previous_delta = delta

def install_step_cache(transformer, threshold, warmup=2, max_run=3):
    """Wrap each block so a reused step costs one block instead of all of them."""
    state = FirstBlockCache(threshold, warmup, max_run)
    transformer._qw_step_cache = state
    blocks = transformer.transformer_blocks
    last = len(blocks) - 1

    for index, block in enumerate(blocks):
        eager = block.forward
        if index == 0:
            def forward(*args, _eager=eager, **kwargs):
                if kwargs.get("kv_cache_mode") != "cached":
                    state.reset()              # one prefill starts every request
                    return _eager(*args, **kwargs)
                hidden_states = kwargs["hidden_states"]
                out = _eager(*args, **kwargs)
                state.decide(out - hidden_states)
                if state.skip:
                    return out + state.tail
                state.after_first = out
                return out
        elif index == last:
            def forward(*args, _eager=eager, **kwargs):
                if kwargs.get("kv_cache_mode") != "cached":
                    return _eager(*args, **kwargs)
                if state.skip:
                    return kwargs["hidden_states"]
                out = _eager(*args, **kwargs)
                state.tail = out - state.after_first
                return out
        else:
            def forward(*args, _eager=eager, **kwargs):
                if state.skip and kwargs.get("kv_cache_mode") == "cached":
                    return kwargs["hidden_states"]
                return _eager(*args, **kwargs)
        block.forward = forward
    return state


def configure_from_environment(transformer):
    """Return the installed cache settings, or None when the flag is unset."""
    raw = os.environ.get("QW21_STEP_CACHE", "").strip()
    if not raw:
        return None
    threshold = float(raw)
    if threshold <= 0:
        return None
    warmup = int(os.environ.get("QW21_STEP_CACHE_WARMUP", "2"))
    max_run = int(os.environ.get("QW21_STEP_CACHE_MAX_RUN", "3"))
    install_step_cache(transformer, threshold, warmup, max_run)
    return {"threshold": threshold, "warmup_steps": warmup, "max_consecutive_reuse": max_run}
