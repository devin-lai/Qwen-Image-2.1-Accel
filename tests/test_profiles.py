# SPDX-License-Identifier: CPAL-1.0
# Copyright (c) 2026 Devin Lai. See LICENSE and NOTICE for attribution terms.
"""CPU regression checks for package imports, profile routing and CLI entry points."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from qwen_image_accel import PROFILES, api
from qwen_image_accel.profiles import get_profile

ROOT = Path(__file__).resolve().parents[1]


class ProfileTests(unittest.TestCase):
    def test_default_is_exact_and_approximation_is_explicit(self):
        default = get_profile("fused")
        self.assertTrue(default.fused)
        self.assertFalse(default.quantized or default.sage or default.compile or default.graph)
        self.assertEqual(get_profile("mxfp8_balanced").skip_blocks, "0,1,2,29,30,31")

    def test_unknown_and_retired_profiles_fail_before_loading(self):
        for name in ("unknown", "fp8", "sage", "fused_cudnn"):
            with self.subTest(name=name), patch.dict(sys.modules, {"torch": None, "diffusers": None}):
                with self.assertRaisesRegex(ValueError, "Unknown|retired"):
                    api.Init('{"profile": "' + name + '"}')

    def test_import_and_module_help_need_no_gpu_dependencies(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        code = "import sys; sys.modules.update(torch=None, triton=None, diffusers=None); import qwen_image_accel; from qwen_image_accel.cli import main; main()"
        result = subprocess.run([sys.executable, "-c", code, "--help"], env=env,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in PROFILES:
            self.assertIn(name, result.stdout)

    def test_launcher_works_outside_checkout(self):
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(["bash", str(ROOT / "run.sh"), "--help"], cwd=cwd,
                                    env={**os.environ, "PYTHON": sys.executable},
                                    text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--profile", result.stdout)

    def test_cli_rejects_unknown_profile(self):
        result = subprocess.run([sys.executable, "-m", "qwen_image_accel", "--profile", "missing", "--prompt", "test"],
                                env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)


if __name__ == "__main__":
    unittest.main()
