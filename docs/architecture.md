# Architecture and API

Read the code in this order:

1. [`profiles.py`](../src/qwen_image_accel/profiles.py) defines supported behavior.
2. [`api.py`](../src/qwen_image_accel/api.py) loads the model and handles requests.
3. [`pipeline.py`](../src/qwen_image_accel/pipeline.py) selects and installs patches.
4. [`optimizations/`](../src/qwen_image_accel/optimizations/) implements those policies.
5. [`kernels/`](../src/qwen_image_accel/kernels/) implements GPU arithmetic.

The package root lazily exposes `configure_pipeline`; importing the package or
requesting CLI help does not import PyTorch or Triton. All internal imports use
the package namespace. No application needs to add a plugin directory to
`sys.path` or import a generic module named `optimization`.

`configure_pipeline(pipe, profile="fused")` accepts a fresh BF16 Diffusers
`QwenImage21Pipeline`. It returns a configuration dictionary, also stored as
`pipe._qw_optimization`. Configuration is a one-time operation: use a new
pipeline to switch profiles. The baseline follows the original initialization.
Other profiles remove unused encoder vocabulary-logit work and broadcast decode
modulation, then install the selected numerical and execution optimizations.

The two decode implementations remain separate deliberately:

- `optimizations/fusion.py` preserves the validated default BF16 route.
- `optimizations/quantized_decode.py` shares activation quantization across
  compatible projections and optionally dispatches SageAttention2.

They share the kernels in `kernels/`, including gated residuals, rotary
embeddings and normalization. `kernels/mxfp8.py` owns the quantized tensor format
and scale layout; `optimizations/mxfp8.py` owns module replacement and coverage.
`compile.py`, `cuda_graph.py`, `step_cache.py` and `vae.py` each own one optional
execution or memory policy. The native request-local prefix KV cache remains
BF16 in every profile; approximate step caching is separate and opt-in.

## JSON adapter

`Init(config="")` loads weights once. `config` is a JSON object with optional
`model` and `profile` fields, which override `QW21_MODEL_PATH` and `QW21_PROFILE`.
An invalid or retired profile is rejected before loading weights.

`Process(body, extra="")` accepts this JSON structure:

```json
{
  "parameter": {
    "prompt": "A red ceramic teapot", "seed": 42, "steps": 40,
    "width": 1024, "height": 1024, "output": "outputs/teapot.png"
  },
  "media_info_list": [{"media_data": "path/to/optional-reference.png"}]
}
```

Omit `media_info_list` for text-to-image generation. Entries may also be HTTP(S)
URLs. For editing, reference images determine dimensions. `extra` is retained
for adapter compatibility and is currently unused.

The return value is `(result_json, status_code, error_message)`. On success,
status is `0` and `result_json.media_info_list[0].media_data` is the PNG path.
Failures return status `20301` and a generic message; details are logged.
Generation and saving are serialized because scheduler, offload and graph state
are mutable. Initialize before accepting requests. Separate processes own
separate pipelines; direct users of `configure_pipeline` manage concurrency.
