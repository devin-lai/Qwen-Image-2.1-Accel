# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""GPU regression tests against native BF16 normalization, rotary and cat."""
import json
from pathlib import Path
Path("results/kernel_checks").mkdir(parents=True, exist_ok=True)
import torch

from qwen_image_accel.kernels.attention import prepare_attention_inputs
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen

torch.set_num_threads(8);torch.manual_seed(721)
results=[]
with torch.inference_mode():
    nq=RMSNorm(128,eps=1e-6).to('cuda',torch.bfloat16)
    nk=RMSNorm(128,eps=1e-6).to('cuda',torch.bfloat16)
    nq.weight.copy_(torch.randn_like(nq.weight));nk.weight.copy_(torch.randn_like(nk.weight))
    for b,s,p in [(1,49,0),(2,257,7),(1,1024,99),(1,4096,113),(1,16384,113),(2,49,1024)]:
        for scale in [0.001,1.,20.]:
            q=torch.randn(b,s,32,128,device='cuda',dtype=torch.bfloat16)*scale
            k=torch.randn_like(q)*scale;v=torch.randn_like(q)
            ck=torch.randn(b,p,32,128,device='cuda',dtype=torch.bfloat16);cv=torch.randn_like(ck)
            phase=torch.randn(s,64,device='cuda');f=torch.polar(torch.ones_like(phase),phase)
            def ref():
                rq=apply_rotary_emb_qwen(nq(q),f,use_real=False)
                rk=apply_rotary_emb_qwen(nk(k),f,use_real=False)
                return rq,torch.cat([ck,rk],1),torch.cat([cv,v],1)
            y=prepare_attention_inputs(q,k,v,nq,nk,f,ck,cv)
            equal=[torch.equal(a,z) for a,z in zip(ref(),y)]
            assert all(equal),(b,s,p,scale,equal)
            results.append(dict(batch=b,seq=s,prefix=p,scale=scale,qkv_equal=equal))
    saved=[x.clone() for x in y]
    ck.add_(1);cv.mul_(2)
    changed=prepare_attention_inputs(q,k,v,nq,nk,f,ck,cv)
    assert all(torch.equal(a,z) for a,z in zip(ref(),changed))
    assert all(torch.equal(a,z) for a,z in zip(saved,y))
    results.append(dict(test='changed_prefix_and_previous_outputs',passed=True))
    assert prepare_attention_inputs(q,k,v,nq,nk,None,ck,cv) is None
    assert prepare_attention_inputs(q.float(),k.float(),v.float(),nq,nk,f,ck,cv) is None
    assert prepare_attention_inputs(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),nq,nk,f,ck,cv) is None
    assert prepare_attention_inputs(q,k,v,nq,nk,f,ck[:1],cv[:1]) is None
    results.append(dict(test='unsupported_input_fallbacks',passed=True))
Path('results/kernel_checks/attention_kernel_tests.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
