# Installation

Use an isolated Python environment. The recorded runtime is Linux, Python 3.10,
PyTorch 2.11.0+cu130, Triton 3.6.0 and RTX 5090 / compute capability 12.0.
Install the matching CUDA/PyTorch environment yourself, plus a C/C++ compiler
and Python development headers for Triton and optional CUDA extension builds.
The scripts never install system packages or model weights.

From a checkout, run `bash scripts/install.sh`. It uses `${PYTHON:-python3}`,
verifies the Diffusers wheel against [the manifest](../third_party/manifest.json),
installs it, and runs `pip install -e` for this project. Paths are resolved from
the script location, so it works from another directory as well.

The package declares its Python dependencies in [pyproject.toml](../pyproject.toml).
PyTorch must already be installed; the fused runtime guard checks the exact
validated PyTorch, CUDA and Triton versions. The pinned Diffusers development
wheel contains the Qwen-Image-2.1 implementation used in the measurements.
A same-version wheel from elsewhere may contain different source. The default
profile checks the transformer source hash and rejects that mismatch.

For a non-editable installation, first install the bundled wheel, then run
`python -m pip install .`. Project wheels and source distributions contain only
the library and license files; obtain the pinned runtime inputs from the Git
checkout before installing a built distribution. The project does not silently
replace them with an arbitrary public Diffusers build.

## Optional SageAttention2

Run `bash scripts/install_sageattention.sh` on the validated machine. It checks
the archive checksum and runtime, builds revision
`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, and installs into the active Python
environment. It uses a temporary build directory and removes it when finished.
Set `MAX_JOBS` to limit parallel compilation. No `vendor/` or `PYTHONPATH`
modification is necessary after installation.

## Checkpoint and outputs

Set `QW21_MODEL_PATH` or pass `--model`. The default path is
`models/Qwen-Image-2.1`, relative to the working directory. Loading is local-only;
obtain the checkpoint under its own terms. Generation defaults to 2048 × 2048,
40 steps and seed 42. Pass a smaller size for a quicker first run.

`safe` avoids the validated fused kernels on other compatible CUDA environments;
it still requires a working Qwen-Image-2.1 Diffusers pipeline and enough memory.
It is not a CPU or Apple Metal inference mode.

## CPU-only development

The adapter, CLI help, profile registry and packaging can be checked without
CUDA, Triton, Diffusers or model weights:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps -e .
python -m pip install Pillow
python -m unittest discover -s tests -v
qwen-image-accel --help
```

`--no-deps` is for development checks only. GPU inference requires the full
runtime installation. For a checkout without installing the package, use
`PYTHONPATH=src python -m unittest discover -s tests -v` or `bash run.sh --help`.
