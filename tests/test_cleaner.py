import unittest

import pandas as pd

from src.cleaner import DataCleaner


class DataCleanerTests(unittest.TestCase):
    def test_category_column_accepts_constant_fill_for_constant_and_mode(self):
        cases = {
            "constant": pd.Series(["a", None], dtype="category"),
            "mode": pd.Series([None, None], dtype="category"),
        }
        for strategy, values in cases.items():
            with self.subTest(strategy=strategy):
                result = DataCleaner(pd.DataFrame({"category": values})).handle_missing(
                    categorical_strategy=strategy, categorical_constant="Unknown",
                )

                self.assertEqual(result["category"].iloc[-1], "Unknown")
                self.assertEqual(str(result["category"].dtype), "category")


if __name__ == "__main__":
    unittest.main()
