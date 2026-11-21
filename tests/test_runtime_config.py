import unittest

from src.runtime_config import detect_cloud_runtime


class RuntimeConfigTests(unittest.TestCase):
    def test_explicit_runtime_overrides_platform_detection(self):
        self.assertTrue(detect_cloud_runtime("cloud", "Windows"))
        self.assertFalse(detect_cloud_runtime("local", "Linux"))

    def test_linux_defaults_to_cloud_and_windows_to_local(self):
        self.assertTrue(detect_cloud_runtime("", "Linux"))
        self.assertFalse(detect_cloud_runtime("", "Windows"))


if __name__ == "__main__":
    unittest.main()
