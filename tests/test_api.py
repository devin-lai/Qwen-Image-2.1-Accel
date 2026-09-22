# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""CPU-only checks for the standalone adapter; no model weights are required."""
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image


from qwen_image_accel import api


class StandaloneAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "nested" / "output.png"
        self.generated = Image.new("RGB", (16, 16), "red")
        self.pipeline = Mock(return_value=types.SimpleNamespace(images=[self.generated]))
        self.torch = types.SimpleNamespace(
            bfloat16="bfloat16", set_num_threads=Mock(), Generator=Mock()
        )
        self.patches = patch.dict("sys.modules", {"torch": self.torch})
        self.patches.start()
        self.addCleanup(self.patches.stop)
        api.pipe = self.pipeline
        self.addCleanup(setattr, api, "pipe", None)

    def test_init_uses_public_pipeline_and_environment(self):
        loader = Mock(return_value=self.pipeline)
        configure = Mock(return_value={"profile": "fused"})
        modules = {
            "diffusers": types.SimpleNamespace(
                QwenImage21Pipeline=types.SimpleNamespace(from_pretrained=loader)
            ),
            "qwen_image_accel.pipeline": types.SimpleNamespace(configure_pipeline=configure),
        }
        with patch.dict("sys.modules", modules), patch.dict(os.environ, {
            "QW21_MODEL_PATH": "models/test-checkpoint",
            "QW21_PROFILE": "fused", "QW21_CPU_THREADS": "4",
        }):
            api.Init()
            loader.assert_called_once_with(
                "models/test-checkpoint", torch_dtype="bfloat16", local_files_only=True
            )
            configure.assert_called_once_with(self.pipeline, "fused")
            self.torch.set_num_threads.assert_called_once_with(4)
            api.Init(json.dumps({"model": "models/override", "profile": "baseline"}))
            self.assertEqual(loader.call_args.args[0], "models/override")
            self.assertEqual(configure.call_args.args[1], "baseline")

    def test_generation_writes_png_and_returns_json_tuple(self):
        result, code, message = api.Process(json.dumps({"parameter": {
            "prompt": "A teapot", "seed": 7, "steps": 3,
            "width": 512, "height": 768, "output": str(self.output),
        }}))
        self.assertEqual((code, message), (0, ""))
        media = json.loads(result)["media_info_list"][0]
        self.assertEqual(media["media_data"], str(self.output))
        self.assertEqual(media["media_profiles"]["media_data_type"], "png")
        with Image.open(self.output) as image:
            self.assertEqual(image.format, "PNG")
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual((kwargs["width"], kwargs["height"]), (512, 768))
        self.assertEqual(kwargs["num_inference_steps"], 3)
        self.assertTrue(kwargs["use_kv_cache"])
        self.torch.Generator.return_value.manual_seed.assert_called_once_with(7)

    def test_editing_keeps_image_dimensions_for_local_and_http_inputs(self):
        source = Path(self.temp.name) / "input.png"
        self.generated.save(source)
        content = BytesIO()
        self.generated.save(content, format="PNG")
        with patch.object(api, "urlopen", return_value=BytesIO(content.getvalue())):
            ok, output = api.algorithm_process({
                "parameter": {"prompt": "Change the background", "output": str(self.output)},
                "media_info_list": [
                    {"media_data": str(source)},
                    {"media_data": "https://example.com/input.png"},
                ],
            })
        self.assertTrue(ok)
        self.assertEqual(output, str(self.output))
        kwargs = self.pipeline.call_args.kwargs
        self.assertEqual(len(kwargs["image"]), 2)
        self.assertEqual(kwargs["image"][0].size, (16, 16))
        self.assertNotIn("width", kwargs)
        self.assertNotIn("height", kwargs)

    def test_failures_return_error_without_exposing_exception_details(self):
        self.pipeline.side_effect = RuntimeError("model diagnostic")
        with self.assertLogs(api.logger, level="ERROR"):
            result, code, message = api.Process(json.dumps({
                "parameter": {"output": str(self.output)}
            }))
        self.assertEqual((json.loads(result), code, message), ({}, 20301, "Image generation failed"))
        self.assertFalse(self.output.exists())
        with self.assertLogs(api.logger, level="ERROR"):
            self.assertEqual(api.Process("invalid JSON")[1], 20301)


if __name__ == "__main__":
    unittest.main()
