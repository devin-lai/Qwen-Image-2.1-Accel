# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Compare paired images and final latents; never equates similarity with quality."""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def ssim_uniform7(a, b):
    """Standard SSIM: 7x7 uniform window, sample covariance, valid image crop."""
    def box(x):
        integral=np.pad(x,((1,0),(1,0),(0,0))).cumsum(0).cumsum(1)
        return (integral[7:,7:]-integral[:-7,7:]-integral[7:,:-7]+integral[:-7,:-7])/49
    ux,uy=box(a),box(b)
    vx,vy=(49/48)*(box(a*a)-ux*ux),(49/48)*(box(b*b)-uy*uy)
    cov=(49/48)*(box(a*b)-ux*uy)
    return float(np.mean(((2*ux*uy+0.01**2)*(2*cov+0.03**2))/
                         ((ux*ux+uy*uy+0.01**2)*(vx+vy+0.03**2))))


def compare(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    rows=[]
    for p in sorted(candidate.glob('*.png')):
        q=reference/p.name
        if not q.exists():
            continue
        a=np.asarray(Image.open(q)).astype(np.float64)/255
        b=np.asarray(Image.open(p)).astype(np.float64)/255
        if a.shape!=b.shape:
            raise ValueError(f'Shape mismatch: {q} {p}')
        mse=float(np.mean((a-b)**2))
        row={'file':p.name,'image_equal':bool(np.array_equal(a,b)),
             'psnr_db':float(-10*np.log10(mse)) if mse else None,
             'mae_0_1':float(np.mean(np.abs(a-b))), 'max_abs_0_1':float(np.max(np.abs(a-b)))}
        row['ssim']=ssim_uniform7(a,b)
        row['channels']=a.shape[-1]
        row['ssim_method']='7x7 uniform, sample covariance, valid crop, data_range=1'
        rgb_mse=float(np.mean((a[...,:3]-b[...,:3])**2))
        row['rgb_psnr_db']=float(-10*np.log10(rgb_mse)) if rgb_mse else None
        row['rgb_ssim']=ssim_uniform7(a[...,:3],b[...,:3])
        if a.shape[-1]==4:
            row['alpha_max_abs_0_1']=float(np.max(np.abs(a[...,3]-b[...,3])))
        if q.with_suffix('.pt').exists() and p.with_suffix('.pt').exists():
            x=torch.load(q.with_suffix('.pt'),map_location='cpu',weights_only=True).double()
            y=torch.load(p.with_suffix('.pt'),map_location='cpu',weights_only=True).double()
            row['latent_equal']=torch.equal(x,y)
            row['latent_relative_rmse']=float((torch.mean((x-y)**2)/torch.mean(x**2)).sqrt())
            row['latent_cosine']=float(torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0))
            row['latent_max_abs']=float((x-y).abs().max())
        rows.append(row)
    if not rows:
        raise ValueError('No matching image pairs; comparison is not evidence')
    def report_path(path):
        # Keep local usernames and absolute workstation paths out of reports.
        try:
            return str(path.resolve().relative_to(Path(__file__).resolve().parents[1]))
        except ValueError:
            return path.name
    return {'reference':report_path(reference),'candidate':report_path(candidate),'pairs':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('--output');a=p.parse_args()
    result=compare(a.reference,a.candidate)
    s=json.dumps(result,indent=2,allow_nan=False)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(s)
    print(s)
