import unittest
from io import BytesIO

from src.file_utils import file_extension, upload_identity


class UploadTests(unittest.TestCase):
    def test_same_name_replacement_loads_new_content(self):
        class Uploaded(BytesIO):
            name = "same.csv"

        first = Uploaded(b"item\nfirst\n")
        second = Uploaded(b"item\nsecond\n")
        self.assertNotEqual(upload_identity(first), upload_identity(second))

    def test_file_extension_is_case_insensitive(self):
        self.assertEqual(file_extension("dataset.CSV"), ".csv")
        self.assertEqual(file_extension("dataset.XLS"), ".xls")


if __name__ == "__main__":
    unittest.main()
