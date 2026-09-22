# Migration from the plugin/experiment layout

Install with `bash scripts/install.sh`, then replace imports from bare `api` or
`optimization` with `from qwen_image_accel import Init, Process, configure_pipeline`.
The JSON request/response contract, `QW21_*` settings and `bash run.sh` remain.
Old internal module paths are removed rather than kept as duplicate packages.

| Previous path | New location |
| --- | --- |
| `plugins/qw_img_21/api.py` | `src/qwen_image_accel/api.py` and `cli.py` |
| `plugins/qw_img_21/optimization.py` | `pipeline.py` and `profiles.py` inside the package |
| `*_ops.py`, MXFP8 kernels | `src/qwen_image_accel/kernels/` |
| Pipeline patches and execution policies | `src/qwen_image_accel/optimizations/` |
| `deps/deps_install.sh` | `scripts/install.sh` |
| `deps/optional/install_sageattention.sh` | `scripts/install_sageattention.sh` |
| Wheel and pinned Sage archive | `third_party/wheels/` and `third_party/sources/` |
| `optimization/benchmarks/` | `benchmarks/` and `tests/gpu/` |
| Round-based documentation | Topic-based `docs/` |
| Raw benchmark evidence | Compact `docs/benchmarks/measurements.json` |

Superseded per-row FP8 profiles are retired explicitly:

| Retired profile | Replacement |
| --- | --- |
| `fp8` | `mxfp8` |
| `fp8_sage`, `fp8_sage_compiled` | `mxfp8_sage` |
| `sage` | `fused_sage` |
| `fused_cudnn` | `fused` with `QW21_ATTENTION=_native_cudnn` |
| `mxfp8_cudnn` | `mxfp8` with `QW21_ATTENTION=_native_cudnn` |

Replacements may change numerical results; retired names fail with migration
advice instead of silently selecting a different implementation. The `fused`
default retains its original kernel arithmetic and provided VAE grid.

The Sage installer now installs into the active environment; a previous
`vendor/` installation is no longer injected by `run.sh`. No Git LFS step is
needed for the retained small dependencies. Full dependency snapshots,
round-specific probes, server automation, generated PNGs/latents and build logs
are removed. Run the current benchmark tools to regenerate outputs in ignored
`results/`; retained JSON preserves historical metrics but cannot substitute for
new image/latent comparisons after changing kernels.
