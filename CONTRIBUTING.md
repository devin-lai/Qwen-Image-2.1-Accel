# Contributing

Start with [installation](docs/installation.md) and
[architecture](docs/architecture.md). Runtime code belongs in
`src/qwen_image_accel`; dependency installation, benchmark orchestration and
historical reports do not belong in the importable package.

For CPU development, install the package with `--no-deps` and install Pillow.
Run `python -m unittest discover -s tests -v` for adapter, profile and entry-point
checks. `python scripts/verify_assets.py` checks both pinned archives.

On the validated CUDA runtime, run `python tests/gpu/run.py`. These checks are
explicit scripts so CPU test discovery never imports Triton or allocates GPU
memory. Use the [benchmark workflow](docs/benchmarks/README.md) to validate
end-to-end numerical behavior after changing kernels or profile composition.
A CPU test pass alone cannot establish BF16 equivalence or GPU performance.

Add supported profiles to the shared registry, document their numerical
tradeoffs, and test configuration before loading large models. Keep approximate
arithmetic opt-in. Preserve native prefill and request-local cache semantics.
For a new kernel, validate against a native reference with boundary sizes,
unsupported-input fallbacks and relevant BF16 rounding cases.

Keep generated output in ignored `results/` or `outputs/`. Do not commit model
weights, tensors, local environments, build products or server configuration.
Retain only compact, relevant measurement summaries with enough configuration
to reproduce them. Preserve all third-party notices and the project attribution
when redistributing changes under [CPAL-1.0](LICENSE).
