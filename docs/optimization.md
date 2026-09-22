# Optimization settings

The supported profiles are defined in [one registry](../src/qwen_image_accel/profiles.py).
The CLI and pipeline consume that registry; no independent profile lists are
maintained in the adapter or benchmark runner.

| Profile | Changes beyond original initialization | Output |
| --- | --- | --- |
| `baseline` | None | Reference |
| `safe` | Features-only encoder, no autoregressive encoder cache, broadcast modulation | Conservative BF16 |
| `fused` | `safe` plus shared modulation, fused Q/K norm, RoPE, prefix packing, LayerNorm/scale, gated residual and SwiGLU | Exact on the recorded cases |
| `fused_sage` | `fused` plus SageAttention2 decode | Approximate |
| `mxfp8` | Fused decode with block-scaled FP8 linears; blocks 0 and 31 stay BF16 | Approximate |
| `mxfp8_sage` | MXFP8 plus SageAttention2 | Approximate |
| `mxfp8_balanced` | MXFP8 with blocks 0,1,2,29,30,31 kept BF16 | Approximate |
| `mxfp8_balanced_sage` | Balanced MXFP8 plus SageAttention2 | Approximate |
| `compiled` | `safe` plus bounded whole-block compilation | Approximate rounding; experimental |
| `graph` | `fused` plus bounded CUDA Graph replay | Experimental, extra VRAM and capture cost |

Fused, MXFP8 and graph profiles require BF16 transformer weights, PyTorch
2.11.0+cu130, Triton 3.6.0, SM120 and the pinned transformer source hash. They
fail clearly on another runtime. `safe` and `baseline` avoid these custom
kernels but still need compatible Diffusers and CUDA. Profiles cannot be
switched on an existing pipeline.

## Environment settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `QW21_PROFILE` | `fused` | Profile; `--profile` overrides it |
| `QW21_MODEL_PATH` | `models/Qwen-Image-2.1` | Local checkpoint; `--model` overrides it |
| `QW21_CPU_THREADS` | `8` | PyTorch CPU thread count |
| `QW21_MXFP8_SKIP_BLOCKS` | `0,31` or balanced preset | Comma-separated BF16 block indices |
| `QW21_MXFP8_LAYERS` | `all` | `all`, `mlp`, `attention`, or explicit linear names |
| `QW21_ATTENTION` | native backend | Diffusers decode backend override, e.g. `_native_cudnn` |
| `QW21_TEXT_ENCODER` | `leaf` | Leaf offload, `resident`, or `block_<n>` |
| `QW21_VAE_TILING` | See below | `provided`, `adaptive`, or `off` |
| `QW21_COMPILE_MAX_SHAPES` | `2` | Compiled tensor configurations per block |
| `QW21_STEP_CACHE` | disabled | Relative-L1 residual-change threshold |
| `QW21_STEP_CACHE_WARMUP` | `2` | Full decode steps before reuse |
| `QW21_STEP_CACHE_MAX_RUN` | `3` | Maximum consecutive residual reuse |
| `QW21_SAGE_ACCUM` | `fp32+fp32` | SageAttention P·V accumulation |
| `QW21_SAGE_QK_GRAN` | `per_warp` | SageAttention Q/K quantization granularity |

The `baseline` profile intentionally ignores optimization knobs; it always
uses the original encoder offload and provided VAE tile grid. Other profiles
read settings at configuration time. Restart before changing them.

MXFP8 retains projection/embedding/modulation layers, norms, VAE, encoder and
KV cache in BF16. Weights are quantized at startup; activations use per-32-element
scales. The GEMM actually uses the block-scaled tensor cores. Non-multiple token
counts are zero-padded to the 128-row hardware alignment. Balanced presets trade
some speed for less quantization error.

SageAttention2 is restricted to unmasked cached decode. Native attention handles
prefill and masked/padded cases. With Sage enabled, an attention backend override
controls the native fallback, not Sage's unmasked kernel. There is no global
attention monkey patch. Alternate accumulation modes were experimental and can
produce NaNs at large token counts; keep the validated default.

## Memory policies

`provided` keeps the checkpoint's VAE tile grid and is the default for baseline,
safe, fused, graph and compiled. Quantized, Sage and step-cache configurations
default to `adaptive`. An explicit VAE override on non-baseline profiles wins.

Adaptive tiling chooses a grid per decode from free VRAM, using a measured
6.72 KiB per output pixel estimate, a 70% budget and 128-pixel overlap. It retries
with the provided grid after an out-of-memory failure. Its pixels depend on the
grid and thus available memory. A different grid changes results even with BF16
weights. `off` always requests a full decode and can exceed available memory.

A resident encoder avoids repeated streaming with the same arithmetic but needs
about 16.33 GiB. It fits best beside a quantized transformer at smaller sizes.
At 2048², the reduced free memory can force smaller VAE tiles and offset the gain.
Coarser `block_<n>` offload remains available but was slower in the experiments.

## Experimental execution

Compilation admits a bounded number of shapes and uses eager execution for
later unseen shapes. Compiler failures disable compilation for that block;
CUDA execution and out-of-memory errors propagate. Compilation changes rounding.
CUDA Graphs cache shape-specific buffers and refresh request data before replay.
Neither is enabled by the default profile.

Step caching skips computation by reusing the network residual when the first
block changes little. No profile enables it automatically. It resets on prefill,
shape or dtype changes and synchronizes the device once per tested step. This
is a numerical approximation, not a substitute for the native prefix KV cache.

Full pipeline measurements cover batch size 1 and one checkpoint. Some kernel
checks cover batch 2; they do not establish full batched inference correctness.
See [benchmarking](benchmarks/README.md) for the retained evidence and GPU checks.
