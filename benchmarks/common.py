# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Shared fixed prompts and synchronized CUDA stage timings."""
import functools

PROMPTS = {
    "product": 'A studio photograph of a red ceramic teapot on a pale wooden table, a small white label reads "QWEN 2.1", soft window light, detailed ceramic texture.',
    "portrait": "A close-up portrait of an elderly woman with silver hair wearing a blue wool scarf, natural skin texture, warm afternoon light, a garden in the background.",
    "type": 'A clean poster with the large text "HELLO WORLD" and the Chinese characters "美好生活", navy and cream colors, precise legible typography, simple geometric decoration.',
    "edit": 'Keep the red teapot and its "QWEN 2.1" label unchanged. Replace the background with a sunlit garden and pink flowers, realistic photography.',
}


class StageTimer:
    def __init__(self):
        self.events = {}

    def wrap(self, obj, name, label):
        import torch
        original = getattr(obj, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self.events.setdefault(label, []).append((start, end))
            return result
        setattr(obj, name, wrapped)

    def collect(self):
        import torch
        torch.cuda.synchronize()
        result = {k: [a.elapsed_time(b) / 1000 for a, b in pairs] for k, pairs in self.events.items()}
        self.events.clear()
        return result
