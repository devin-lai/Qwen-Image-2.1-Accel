# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Validate the final guarded production LayerNorm fusion, including fallback."""
import json
from pathlib import Path
Path("results/kernel_checks").mkdir(parents=True, exist_ok=True)
import torch

from qwen_image_accel.kernels.layernorm import layer_norm_scale as welford_scale
from helpers import bench, difference

if __name__=='__main__':
    torch.manual_seed(717);torch.set_num_threads(8);results=[]
    norm=torch.nn.LayerNorm(4096,eps=1e-6,elementwise_affine=False).cuda()
    with torch.inference_mode():
        for b,n in [(1,49),(2,257),(1,1024),(1,4096),(1,16384)]:
            for scale in [0.001,1.,20.]:
                x=torch.randn(b,n,4096,device='cuda',dtype=torch.bfloat16)*scale
                s=torch.randn(b,1,4096,device='cuda',dtype=torch.bfloat16)
                t,ref=bench(lambda:norm(x)*s)
                t2,y=bench(lambda:welford_scale(x,s,norm))
                row=dict(batch=b,seq=n,scale=scale,old_ms=t,new_ms=t2,difference=difference(ref,y))
                assert row['difference']['equal'], row
                results.append(row);print(json.dumps(row),flush=True)
    Path('results/kernel_checks/layernorm_kernel_tests.json').write_text(json.dumps(results,indent=2))

    # CPU/non-BF16, affine and non-contiguous paths use native LayerNorm.
    for device,dtype,affine,strided in [('cpu',torch.float32,False,False),('cuda',torch.bfloat16,True,False),('cuda',torch.bfloat16,False,True)]:
        x=torch.randn(2,49,8192 if strided else 4096,device=device,dtype=dtype)
        if strided:x=x[...,::2]
        scale=torch.randn(2,1,4096,device=device,dtype=dtype)
        norm=torch.nn.LayerNorm(4096,eps=1e-6,elementwise_affine=affine).to(device,dtype)
        assert torch.equal(norm(x)*scale,welford_scale(x,scale,norm))
    results.append(dict(test='unsupported_input_fallbacks',passed=True))
    Path('results/kernel_checks/layernorm_kernel_tests.json').write_text(json.dumps(results,indent=2))
