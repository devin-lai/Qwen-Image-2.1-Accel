# Contributing

Documentation improvements, reproducible bug reports, GPU validation and
measured optimizations are all welcome. You do not need a GPU to improve the
docs or run the CPU test suite.

## Report a problem or share a result

Use the [issue forms](https://github.com/devin-lai/Qwen-Image-2.1-Accel/issues/new/choose)
for bugs, questions, feature requests and benchmarks. Search existing issues
first. Include the commit, GPU/VRAM, OS, Python/PyTorch/CUDA/Triton versions,
profile, relevant `QW21_*` settings and the smallest command that reproduces
the behavior. Remove credentials and private paths from logs before posting.

For performance reports, include warmup, timed repeats, image size, steps,
seed, timing boundaries and a baseline comparison. Attach compact JSON reports
and note any accuracy differences; follow the
[benchmark workflow](docs/benchmarks/README.md). Results from unvalidated
hardware are useful observations, not proof of supported inference.

## Make a change

Keep pull requests focused and explain the user-visible behavior and validation.
For documentation changes, check commands against `--help` and verify links.
For runtime changes, use the checks below and record the tested configuration.

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
