# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Compiler wrappers must not hide CUDA execution or allocation failures."""
import json
from pathlib import Path
Path("results/kernel_checks").mkdir(parents=True, exist_ok=True)
import torch
from torch._dynamo.exc import BackendCompilerFailed

from qwen_image_accel.optimizations.compile import BoundedCompile

def eager(x): return x+1
rows=[]
for inner in [torch.OutOfMemoryError('allocation sentinel'),
              RuntimeError('CUDA error: an illegal memory access was encountered'),
              RuntimeError('CUBLAS_STATUS_EXECUTION_FAILED')]:
    guard=BoundedCompile(eager)
    def fail(*args): raise BackendCompilerFailed(eager,inner,None)
    guard.compiled=fail
    try: guard(torch.ones(1))
    except BackendCompilerFailed:
        assert not guard.disabled and guard.eager_calls==0
        rows.append(dict(error=str(inner),propagated=True))
    else: raise AssertionError('Wrapped GPU error incorrectly retried')
Path('results/kernel_checks/compiler_error_propagation.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
