import importlib
import os
import unittest
from unittest.mock import patch

import src.runtime_config as runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_explicit_runtime_overrides_platform_detection(self):
        self.assertTrue(runtime_config.detect_cloud_runtime("cloud", "Windows"))
        self.assertFalse(runtime_config.detect_cloud_runtime("local", "Linux"))

    def test_linux_defaults_to_cloud_and_windows_to_local(self):
        self.assertTrue(runtime_config.detect_cloud_runtime("", "Linux"))
        self.assertFalse(runtime_config.detect_cloud_runtime("", "Windows"))

    def test_runtime_profiles_apply_expected_limits(self):
        try:
            with patch.dict(os.environ, {"VIZML_RUNTIME": "cloud"}):
                importlib.reload(runtime_config)
                self.assertEqual(runtime_config.MAX_UPLOAD_MB, 100)
                self.assertEqual(runtime_config.MAX_TRAINING_ROWS, 50_000)
                self.assertEqual(runtime_config.MAX_MODELS, 4)
                self.assertEqual(runtime_config.MAX_CV_FOLDS, 5)

            with patch.dict(os.environ, {"VIZML_RUNTIME": "local"}):
                importlib.reload(runtime_config)
                self.assertEqual(runtime_config.MAX_UPLOAD_MB, 200)
                self.assertIsNone(runtime_config.MAX_TRAINING_ROWS)
                self.assertIsNone(runtime_config.MAX_MODELS)
                self.assertEqual(runtime_config.MAX_CV_FOLDS, 10)
        finally:
            importlib.reload(runtime_config)


if __name__ == "__main__":
    unittest.main()
