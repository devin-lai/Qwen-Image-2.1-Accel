# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Bound optional block compilation and retain an eager route for new shapes."""
import logging
import torch
from torch._dynamo.exc import BackendCompilerFailed, FailOnRecompileLimitHit, Unsupported

log = logging.getLogger(__name__)


def _signature(value):
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
    if isinstance(value, (tuple, list)):
        return tuple(_signature(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, _signature(v)) for k, v in sorted(value.items()))
    if hasattr(value, 'k') and hasattr(value, 'v'):
        return (_signature(value.k), _signature(value.v))
    if isinstance(value, slice):
        return (value.start, value.stop, value.step)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return type(value)


class BoundedCompile:
    """Admit at most max_shapes tensor configurations per block per process.

    Unseen configurations after the budget execute eagerly. A compiler failure
    disables this block's compiled route; genuine CUDA execution errors propagate.
    Model weights and the native request-local cache are shared by both routes.
    """
    def __init__(self, eager, max_shapes=2):
        if max_shapes < 0:
            raise ValueError('QW21_COMPILE_MAX_SHAPES must be nonnegative')
        self.eager = eager
        self.max_shapes = max_shapes
        self.shapes = set()
        self.disabled = False
        self.compiled_calls = self.eager_calls = self.compiler_failures = 0
        self.compiled = torch.compile(eager, fullgraph=True, dynamic=False,
                                      options={'triton.cudagraphs': False}) if max_shapes else eager

    def __call__(self, *args, **kwargs):
        if self.disabled or self.max_shapes == 0:
            self.eager_calls += 1
            return self.eager(*args, **kwargs)
        signature = (_signature(args), _signature(kwargs))
        if signature not in self.shapes and len(self.shapes) >= self.max_shapes:
            self.eager_calls += 1
            return self.eager(*args, **kwargs)
        self.shapes.add(signature)
        try:
            result = self.compiled(*args, **kwargs)
            self.compiled_calls += 1
            return result
        except (BackendCompilerFailed, FailOnRecompileLimitHit, Unsupported) as exc:
            # A compiler wrapper can contain an allocation failure. Do not mask it.
            cause, visited = exc, set()
            while cause is not None and id(cause) not in visited:
                visited.add(id(cause))
                message = str(cause).lower()
                cuda_execution_failure = any(token in message for token in (
                    'cuda error:', 'device-side assert', 'illegal memory access',
                    'cublas_status_execution_failed', 'cudnn_status_execution_failed'))
                if isinstance(cause, torch.OutOfMemoryError) or cuda_execution_failure:
                    raise
                cause = cause.__cause__ or cause.__context__ or getattr(cause, 'inner_exception', None)
            self.disabled = True
            self.compiler_failures += 1
            self.eager_calls += 1
            log.warning('Qwen block compilation disabled after %s; using eager execution', type(exc).__name__)
            return self.eager(*args, **kwargs)

    def stats(self):
        return dict(admitted_shapes=len(self.shapes), max_shapes=self.max_shapes,
                    disabled=self.disabled, compiled_calls=self.compiled_calls,
                    eager_calls=self.eager_calls, compiler_failures=self.compiler_failures)
