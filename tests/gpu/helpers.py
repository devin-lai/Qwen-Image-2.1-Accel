# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Shared GPU timing and difference helpers."""
import torch


def bench(fn, repeats=10):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats): y = fn()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end)/repeats, y

def difference(a, b):
    diff = a.float()-b.float()
    return dict(equal=torch.equal(a,b), max_abs=float(diff.abs().max()),
                rel_rmse=float((diff.square().mean()/a.float().square().mean()).sqrt()),
                mismatch=float((a!=b).float().mean()))

