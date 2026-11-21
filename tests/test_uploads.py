import unittest
import csv
from io import BytesIO, StringIO

import pandas as pd

from src.file_utils import dataframe_to_safe_csv, file_extension, upload_identity


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

    def test_csv_export_neutralizes_formula_text_without_changing_source(self):
        dataframe = pd.DataFrame({"=header": ["=1+1", "plain"], "number": [-2, 3]})

        rows = list(csv.reader(StringIO(dataframe_to_safe_csv(dataframe))))

        self.assertEqual(rows[0][0], "\t=header")
        self.assertEqual(rows[1], ["\t=1+1", "-2"])
        self.assertEqual(dataframe.iloc[0, 0], "=1+1")


if __name__ == "__main__":
    unittest.main()
