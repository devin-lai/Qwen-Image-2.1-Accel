# Measurements and reproduction

[measurements.json](measurements.json) preserves the successful baseline and
current-profile reports, pairwise comparisons and teacher-forced numerical
records from 2026-09-21/22. Paths and implementation hashes inside the records
refer to the original experiment layout. They are provenance fields, not links
to files still shipped in this repository.

The records predate the package refactor. Full-resolution PNGs, final latent
tensors, build logs, failed runs and superseded probes were removed to keep the
checkout small. Their exclusion means comparisons cannot be recomputed from
these records alone; regenerate images and tensors with the workflow below.

## Recorded latency

Linux, RTX 5090 (32 GiB, SM120), Python 3.10, PyTorch 2.11.0+cu130, Triton 3.6.0,
Transformers 5.17.0 and the pinned Diffusers wheel. The GPU was otherwise idle.
Each latency is the mean of two product-prompt requests at 40 steps, seed 42,
batch size 1, following a separate two-step warmup for that resolution. Timings
include encoding, denoising and VAE decode; loading, saving and network time are
excluded. Stage timings use synchronized CUDA events.

| Configuration | 512² | 1024² | 2048² |
| --- | ---: | ---: | ---: |
| `baseline` | 3.97 s | 15.31 s | 82.47 s |
| `fused` | 3.56 s | 13.89 s | 74.07 s |
| `fused` + adaptive VAE | 3.52 s | 13.68 s | 73.30 s |
| `fused_sage` | 3.49 s | 12.77 s | 58.12 s |
| `mxfp8_sage` | 1.83 s | 6.48 s | 33.67 s |
| `mxfp8_sage` + resident encoder | 1.55 s | 6.27 s | 33.57 s |

Sage and quantized configurations use adaptive VAE tiling by default. The
resident-encoder row sets `QW21_TEXT_ENCODER=resident`. No row enables step
caching, whole-block compilation or CUDA Graphs. Startup, new-shape compilation
and graph capture costs must be considered separately when deploying.

## Recorded accuracy

Twelve image/latent pairs per configuration: two product runs, one portrait and
one typography prompt at each of three resolutions. All use the same seeds and
steps as the baseline. A null PSNR in the JSON denotes zero error (infinite PSNR).

| Configuration | Mean RGB PSNR | Minimum RGB SSIM | Exact image and latent pairs |
| --- | ---: | ---: | ---: |
| `fused` | Infinite | 1.00000 | 12 / 12 |
| `fused` + adaptive VAE | 38.51 dB | 0.97201 | 0 / 12 |
| `fused_sage` | 34.18 dB | 0.90164 | 0 / 12 |
| `mxfp8_sage` | 30.29 dB | 0.93355 | 0 / 12 |
| `mxfp8_sage` + resident encoder | 30.34 dB | 0.93400 | 0 / 12 |

PSNR/SSIM describe similarity, not perceptual quality. SSIM uses a 7 × 7 uniform
window, sample covariance, valid crop and data range 1. RGB and alpha are
reported separately. Latent comparison includes exact equality, relative RMSE
and cosine similarity. A sampler can amplify small arithmetic differences over
40 steps, so image divergence alone does not measure per-step numerical error.

Teacher-forced replay instead records the baseline trajectory and supplies its
latents to the candidate after each scheduler step. Velocity-prediction error
then measures numerical differences without the candidate's sampler feedback.
The historical `fused` replay returns zero error on its recorded cases.

These are results for one checkpoint, batch size 1 and the recorded workloads.
They do not prove equivalence for every prompt, mask, editing case or runtime.

## Reproduce on the validated GPU

Install the runtime and package first. Run with no other GPU workloads:

```bash
export QW21_MODEL_PATH=/path/to/Qwen-Image-2.1
python tests/gpu/run.py

python benchmarks/run.py --profile baseline --quality --output results/baseline
python benchmarks/run.py --profile fused --quality --output results/fused
python benchmarks/compare.py results/baseline results/fused \
  --output results/fused-comparison.json

# Optional: requires scripts/install_sageattention.sh first.
python benchmarks/run.py --profile mxfp8_sage --quality --output results/mxfp8_sage
python benchmarks/compare.py results/baseline results/mxfp8_sage \
  --output results/mxfp8_sage-comparison.json

python benchmarks/teacher_forced.py --profiles fused mxfp8_sage \
  --sizes 512 1024 2048 --prompts product type --steps 40 \
  --output results/numerical-error.json
```

For a shorter timing run, use `--sizes 512 --repeats 1`. Use `--image input.png`
to benchmark editing with the reference image's dimensions; this runs one edit
case rather than a synthetic resolution sweep. All runners accept a configurable
model path; none depends on a particular remote server or workstation layout.

`benchmarks/run.py` writes `report.json`, PNGs and final latent `.pt` files under
`--output`. Each size has its own warmup. The JSON includes configuration,
package versions, implementation hashes, stage times and peak GPU memory.
`tests/gpu/run.py` runs kernel, fallback, graph request-refresh and compiler-error
checks serially. Run the end-to-end comparisons as well after arithmetic changes.
