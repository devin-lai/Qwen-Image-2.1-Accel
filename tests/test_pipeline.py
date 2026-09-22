# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""Exercise profile composition without loading CUDA dependencies or weights."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from qwen_image_accel import PROFILES

ROOT = Path(__file__).resolve().parents[1]


class PipelineConfigurationTests(unittest.TestCase):
    def enterContext(self, context):
        value = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return value

    def setUp(self):
        self.offload = Mock()
        self.quantize = Mock(return_value=["linear"])
        self.fusion = Mock()
        self.composed = Mock()
        self.graph = Mock()
        self.cache = Mock(return_value=None)
        self.vae = Mock(side_effect=lambda vae, approximate: {
            "policy": os.environ.get("QW21_VAE_TILING") or ("adaptive" if approximate else "provided")
        })
        self.torch = NS(bfloat16="bf16", device=lambda device: device,
                        __version__="2.11.0+cu130", version=NS(cuda="13.0"),
                        cuda=NS(get_device_capability=lambda: (12, 0)))
        modules = {
            "torch": self.torch,
            "diffusers": NS(), "diffusers.hooks": NS(apply_group_offloading=self.offload),
            "sageattention": NS(sageattn_qk_int8_pv_fp8_cuda=Mock()),
            "qwen_image_accel.kernels.bf16": NS(install_qk_norm=Mock(), install_swiglu=Mock()),
        }
        for name, module in {
            "mxfp8": NS(quantize_transformer=self.quantize),
            "fusion": NS(install_fused_decode=self.fusion),
            "quantized_decode": NS(install_mx_fused_decode=self.composed),
            "cuda_graph": NS(install_cuda_graph=self.graph),
            "sage_attention": NS(ACCUM="fp32+fp32", QK_GRAN="per_warp"),
            "step_cache": NS(configure_from_environment=self.cache),
            "vae": NS(configure_vae=self.vae),
        }.items():
            modules["qwen_image_accel.optimizations." + name] = module
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        spec = importlib.util.spec_from_file_location("qwen_image_accel.pipeline", ROOT / "src/qwen_image_accel/pipeline.py")
        self.pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.pipeline)
        self.optimize = self.enterContext(patch.object(self.pipeline, "_optimize_decode_blocks"))

    def fresh_pipe(self):
        return NS(transformer=NS(dtype="bf16", to=Mock(), transformer_blocks=[
            NS(attn=NS(processor=NS(_attention_backend=None)))
        ]), vae=NS(to=Mock(), enable_tiling=Mock()),
                  text_encoder=NS(model=Mock(), forward=Mock()))

    def configure(self, pipe, name):
        with patch.object(self.pipeline, "_validate_fused_runtime"):
            return self.pipeline.configure_pipeline(pipe, name)

    def test_all_supported_profiles_route_to_expected_implementations(self):
        for name, profile in PROFILES.items():
            with self.subTest(profile=name):
                for mock in (self.quantize, self.fusion, self.composed, self.graph, self.cache, self.vae, self.optimize):
                    mock.reset_mock()
                pipe = self.fresh_pipe()
                config = self.configure(pipe, name)
                self.assertIs(config, pipe._qw_optimization)
                self.assertEqual(config["profile"], name)
                self.assertEqual(self.quantize.called, profile.quantized)
                self.assertEqual(self.graph.called, profile.graph)
                self.assertEqual(self.composed.called, profile.quantized or profile.sage)
                self.assertEqual(self.fusion.called, profile.fused and not (profile.quantized or profile.sage))
                if name == "baseline":
                    pipe.vae.enable_tiling.assert_called_once()
                    self.vae.assert_not_called()
                else:
                    self.assertEqual(config["approximate"], profile.quantized or profile.sage or profile.compile)
                    self.vae.assert_called_once_with(pipe.vae, profile.quantized or profile.sage)
                    self.assertEqual(self.optimize.called, not profile.fused)
                    if self.optimize.called:
                        self.optimize.assert_called_once_with(pipe.transformer, compile_blocks=profile.compile)

    def test_balanced_overrides_match_the_configuration_report(self):
        pipe = self.fresh_pipe()
        with patch.dict(os.environ, {"QW21_MXFP8_SKIP_BLOCKS": "0,31", "QW21_MXFP8_LAYERS": "mlp"}):
            config = self.configure(pipe, "mxfp8_balanced")
        self.quantize.assert_called_once_with(pipe.transformer, "0,31", "mlp")
        self.assertEqual(config["mxfp8_skipped_blocks"], "0,31")
        self.assertEqual(config["mxfp8_layer_selection"], "mlp")

    def test_default_stays_unquantized_with_provided_tiling(self):
        config = self.configure(self.fresh_pipe(), "fused")
        self.assertFalse(config["approximate"])
        self.assertEqual(config["vae_tiling"]["policy"], "provided")
        self.quantize.assert_not_called()
        self.composed.assert_not_called()
        self.graph.assert_not_called()

    def test_explicit_vae_and_attention_changes_are_reported(self):
        for env in ({"QW21_VAE_TILING": "adaptive"}, {"QW21_ATTENTION": "_native_cudnn"}):
            with self.subTest(env=env), patch.dict(os.environ, env):
                config = self.configure(self.fresh_pipe(), "fused")
                self.assertTrue(config["approximate"])

    def test_step_cache_remains_opt_in_and_affects_tiling(self):
        self.cache.return_value = {"threshold": 0.06}
        pipe = self.fresh_pipe()
        config = self.configure(pipe, "fused")
        self.assertTrue(config["approximate"])
        self.assertEqual(config["step_cache"], {"threshold": 0.06})
        self.vae.assert_called_once_with(pipe.vae, True)

    def test_configuration_cannot_be_applied_twice(self):
        pipe = self.fresh_pipe()
        self.configure(pipe, "safe")
        with self.assertRaisesRegex(RuntimeError, "fresh pipeline"):
            self.configure(pipe, "fused")

    def test_invalid_profile_has_no_pipeline_side_effects(self):
        pipe = self.fresh_pipe()
        with self.assertRaises(ValueError):
            self.pipeline.configure_pipeline(pipe, "missing")
        pipe.transformer.to.assert_not_called()
        self.offload.assert_not_called()

    def test_runtime_and_dtype_guards_run_before_mutation(self):
        pipe = self.fresh_pipe()
        pipe.transformer.dtype = "float32"
        with self.assertRaisesRegex(ValueError, "BF16"):
            self.pipeline.configure_pipeline(pipe)
        pipe.transformer.dtype = "bf16"
        self.torch.__version__ = "2.0.0"
        with self.assertRaisesRegex(RuntimeError, "unvalidated runtime"):
            self.pipeline.configure_pipeline(pipe)
        pipe.transformer.to.assert_not_called()

    def test_changed_transformer_source_is_rejected(self):
        pipe = self.fresh_pipe()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "transformer.py"
            source.write_text("# changed source\n")
            with patch.dict(sys.modules, {"diffusers.models.transformers": NS(transformer_qwenimage21=NS(__file__=str(source)))}), \
                    patch.object(self.pipeline, "version", return_value="3.6.0"):
                with self.assertRaisesRegex(RuntimeError, "source differs"):
                    self.pipeline.configure_pipeline(pipe)
        pipe.transformer.to.assert_not_called()


if __name__ == "__main__":
    unittest.main()
