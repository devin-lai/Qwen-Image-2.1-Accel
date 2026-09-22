# Qwen-Image-2.1-Accel

Qwen-Image-2.1 inference on one RTX 5090, with a Python API and a command-line
interface. The default `fused` profile preserves BF16 weights and native prefix
KV caching. Quantization, SageAttention2, compilation, CUDA Graphs and step
caching are explicit opt-ins.

## Quick start

The validated environment is **Linux, RTX 5090 (32 GB / SM120), Python 3.10,
PyTorch 2.11.0+cu130 and Triton 3.6.0**. Install CUDA/PyTorch first, obtain model
weights separately, then run:

```bash
git clone https://github.com/devin-lai/Qwen-Image-2.1-Accel.git
cd Qwen-Image-2.1-Accel
bash scripts/install.sh

export QW21_MODEL_PATH=/path/to/Qwen-Image-2.1
qwen-image-accel --prompt "A red ceramic teapot on a wooden table" \
  --width 1024 --height 1024 --output outputs/teapot.png
```

`bash run.sh` and `python -m qwen_image_accel` expose the same CLI.
Use `--help` to list options, `--image input.png` for editing (repeat for multiple
references), or `--profile baseline` for the original initialization. Input
images determine editing dimensions. Outputs are PNG files.

The installer verifies the bundled Diffusers wheel and installs the package in
editable mode. Git LFS is not needed. See [installation](docs/installation.md)
for dependency details and CPU-only development.

## Choose an optimization profile

| Profile | Use |
| --- | --- |
| `fused` (default) | Selective BF16 fusion; bit-identical on the recorded test cases |
| `baseline` | Original BF16 initialization for comparison |
| `safe` | Conservative changes without the validated Triton kernels |
| `fused_sage` | BF16 weights with approximate SageAttention2 attention |
| `mxfp8`, `mxfp8_sage` | Faster block-scaled FP8, with numerical differences |
| `mxfp8_balanced`, `mxfp8_balanced_sage` | Keep six boundary blocks in BF16 |
| `compiled`, `graph` | Experimental execution modes |

To enable the fastest measured profile:

```bash
bash scripts/install_sageattention.sh
qwen-image-accel --profile mxfp8_sage --prompt "A red ceramic teapot" \
  --width 1024 --height 1024 --output outputs/teapot-fast.png
```

[Optimization settings](docs/optimization.md) explain accuracy tradeoffs,
VAE tiling, encoder placement and runtime guards.

## Python integration

```python
import json
from qwen_image_accel import Init, Process

Init(json.dumps({"model": "/path/to/Qwen-Image-2.1", "profile": "fused"}))
result, status, error = Process(json.dumps({
    "parameter": {
        "prompt": "A red ceramic teapot", "width": 1024, "height": 1024,
        "steps": 40, "seed": 42, "output": "outputs/teapot.png",
    }
}))
```

Status `0` means success; `result` is JSON containing the output image path.
Requests are serialized around the shared pipeline. For an existing BF16
`QwenImage21Pipeline`, call `configure_pipeline(pipe, "fused")` once before use.
See the [architecture and API guide](docs/architecture.md).

## Repository map

```text
src/qwen_image_accel/  Installable library
  api.py, cli.py      JSON adapter and command-line interface
  profiles.py         Shared profile definitions
  pipeline.py         Pipeline configuration and compatibility checks
  kernels/            Triton arithmetic and tensor layouts
  optimizations/      Pipeline patches, quantization and execution policies
benchmarks/           Inference timing, numerical replay and comparison tools
tests/                CPU tests; explicit GPU checks under gpu/
scripts/              Dependency installation and checksum verification
docs/                 Setup, architecture, tuning and recorded measurements
third_party/          Only the pinned Diffusers wheel and SageAttention source
licenses/             Preserved upstream license notices
```

Generated images, tensors, checkpoints, build output and caches stay outside
version control. Historical experiment copies and machine-specific scripts have
been removed. Compact [measurement records](docs/benchmarks/measurements.json)
retain the evidence for the published results. See [migration](docs/migration.md)
for old-to-new paths and retired profiles.

## Recorded performance

Historical measurements on the environment above, 40 steps, seed 42, batch 1;
two product-prompt runs per size after a separate two-step warmup. Time includes
text encoding, denoising and VAE decoding, and excludes loading and PNG saving.

| Profile | 512 × 512 | 1024 × 1024 | 2048 × 2048 |
| --- | ---: | ---: | ---: |
| `baseline` | 3.97 s | 15.31 s | 82.47 s |
| `fused` | 3.56 s | 13.89 s | 74.07 s |
| `fused_sage` | 3.49 s | 12.77 s | 58.12 s |
| `mxfp8_sage` | 1.83 s | 6.48 s | 33.67 s |
| `mxfp8_sage` + resident encoder | 1.55 s | 6.27 s | 33.57 s |

`fused` matched all 12 recorded image and final-latent pairs exactly. This is
case-specific evidence, not a universal guarantee. These measurements predate
the package refactor; they are not a new GPU validation. See
[method, accuracy and reproduction commands](docs/benchmarks/README.md).

## Development

```bash
python -m pip install --no-deps -e .  # after runtime setup; see CPU-only setup
python -m unittest discover -s tests -v
python scripts/verify_assets.py
```

The [contributing guide](CONTRIBUTING.md) describes CPU tests, GPU validation and
where to add new optimizations.

## License and attribution

Project code is licensed under [CPAL-1.0](LICENSE), an
[OSI-approved license](https://opensource.org/license/CPAL-1.0). Exhibit B
requires the following attribution in applicable graphical interfaces,
including Larger Works:

> Copyright (c) 2026 Devin Lai · Powered by
> [Qwen-Image-2.1-Accel](https://github.com/devin-lai/Qwen-Image-2.1-Accel)

Section 14 exempts access without a graphical interface from the display
requirement. CPAL also requires source availability for covered code when
distributed or externally deployed, including over a network. It does not
require watermarking generated images. See [licensing](docs/licensing.md) and
[NOTICE](NOTICE); dependencies and model weights retain their own terms.
