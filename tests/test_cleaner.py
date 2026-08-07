import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from src.cleaner import DataCleaner

def test_profile():
    df = pd.DataFrame({
        "A": [1, 2, np.nan],
        "B": ["cat", "dog", "cat"]
    })
    cleaner = DataCleaner(df)
    prof = cleaner.profile()
    
    assert prof["shape"] == [3, 2]
    assert prof["dtypes"]["A"] == "float64"
    # Accept both legacy 'object' and modern pandas 'str'/'string' dtypes
    assert prof["dtypes"]["B"] in ["object", "string", "str"]
    assert prof["null_counts"]["A"] == 1
    assert prof["null_counts"]["B"] == 0
    assert prof["unique_counts"]["B"] == 2
    assert "describe" in prof

def test_handle_missing():
    df = pd.DataFrame({
        "num": [1.0, 3.0, np.nan, 5.0],
        "cat": ["apple", np.nan, "banana", "apple"]
    })
    
    # 1. Median & Mode (defaults)
    cleaner = DataCleaner(df)
    res_default = cleaner.handle_missing()
    # median of [1, 3, 5] is 3.0
    assert res_default["num"].tolist() == [1.0, 3.0, 3.0, 5.0]
    # mode of ['apple', 'banana', 'apple'] is 'apple'
    assert res_default["cat"].tolist() == ["apple", "apple", "banana", "apple"]
    
    # 2. Mean & Constant
    res_custom = cleaner.handle_missing(
        numeric_strategy="mean",
        categorical_strategy="constant",
        categorical_constant="Unknown"
    )
    # mean of [1, 3, 5] is 3.0
    assert res_custom["num"].tolist() == [1.0, 3.0, 3.0, 5.0]
    assert res_custom["cat"].tolist() == ["apple", "Unknown", "banana", "apple"]

    # 3. ffill
    res_ffill = cleaner.handle_missing(
        numeric_strategy="ffill",
        categorical_strategy="ffill"
    )
    # ffill of [1, 3, nan, 5] is [1, 3, 3, 5]
    assert res_ffill["num"].tolist() == [1.0, 3.0, 3.0, 5.0]
    # ffill of ['apple', nan, 'banana', 'apple'] is ['apple', 'apple', 'banana', 'apple']
    assert res_ffill["cat"].tolist() == ["apple", "apple", "banana", "apple"]

    # 4. bfill
    res_bfill = cleaner.handle_missing(
        numeric_strategy="bfill",
        categorical_strategy="bfill"
    )
    # bfill of [1, 3, nan, 5] is [1, 3, 5, 5]
    assert res_bfill["num"].tolist() == [1.0, 3.0, 5.0, 5.0]
    # bfill of ['apple', nan, 'banana', 'apple'] is ['apple', 'banana', 'banana', 'apple']
    assert res_bfill["cat"].tolist() == ["apple", "banana", "banana", "apple"]

def test_coerce_types():
    df = pd.DataFrame({
        "num_str": ["1", "2.5", "3", "bad_num"],      # 75% numeric
        "date_str": ["2020-01-01", "2020-01-02", "not_a_date", "2020-01-04"], # 75% date
        "text_str": ["hello", "world", "foo", "bar"]  # 0% numeric/date
    })
    cleaner = DataCleaner(df)
    res = cleaner.coerce_types()
    
    # Should cast num_str to float/numeric
    assert pd.api.types.is_numeric_dtype(res["num_str"])
    assert pd.isna(res["num_str"].iloc[3])  # 'bad_num' should be NaN
    
    # Should cast date_str to datetime
    assert pd.api.types.is_datetime64_any_dtype(res["date_str"])
    assert pd.isna(res["date_str"].iloc[2])  # 'not_a_date' should be NaN
    
    # Should keep text_str as object/string
    assert pd.api.types.is_object_dtype(res["text_str"]) or pd.api.types.is_string_dtype(res["text_str"])
    
    # Check flags
    assert "num_str" in cleaner.flagged_columns
    assert "date_str" in cleaner.flagged_columns
    assert "text_str" in cleaner.flagged_columns
    assert "Partial numeric" in cleaner.flagged_columns["num_str"]
    assert "Partial datetime" in cleaner.flagged_columns["date_str"]
    assert "Unparseable" in cleaner.flagged_columns["text_str"]

def test_outliers():
    # Outliers based on IQR
    # Data: 1, 2, 3, 4, 100 (100 is clearly an outlier)
    # Q1=2, Q3=4, IQR=2. lower = 2 - 3 = -1, upper = 4 + 3 = 7
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0, 100.0]
    })
    
    cleaner = DataCleaner(df)
    
    # 1. Detect outliers
    mask = cleaner.detect_outliers()
    assert mask["A"].tolist() == [False, False, False, False, True]
    
    # 2. Clip outliers
    res_clip = cleaner.handle_outliers(action="clip")
    assert res_clip["A"].tolist() == [1.0, 2.0, 3.0, 4.0, 7.0]
    
    # 3. Remove outliers
    res_remove = cleaner.handle_outliers(action="remove")
    assert res_remove["A"].tolist() == [1.0, 2.0, 3.0, 4.0]
    
    # 4. Flag outliers
    res_flag = cleaner.handle_outliers(action="flag")
    assert "A_outlier" in res_flag.columns
    assert res_flag["A_outlier"].tolist() == [False, False, False, False, True]

def test_duplicates():
    df = pd.DataFrame({
        "A": [1, 2, 2, 3],
        "B": ["x", "y", "y", "z"]
    })
    cleaner = DataCleaner(df)
    assert cleaner.get_duplicate_count() == 1
    
    res = cleaner.remove_duplicates()
    assert len(res) == 3
    assert res.duplicated().sum() == 0

def test_categorical_encoding():
    df = pd.DataFrame({
        "cat": ["apple", "banana", "apple", np.nan],
        "num": [1, 2, 3, 4]
    })
    
    # 1. One-hot encoding
    cleaner = DataCleaner(df)
    res_oh = cleaner.encode_categoricals(method="onehot")
    assert "cat_apple" in res_oh.columns
    assert "cat_banana" in res_oh.columns
    # NaN might be ignored or handled by dummies depending on pandas config.
    # We check that one-hot columns are integers (0 or 1)
    assert res_oh["cat_apple"].tolist()[:3] == [1, 0, 1]
    
    # 2. Label encoding - build fresh cleaner so df is unmodified
    cleaner2 = DataCleaner(df)
    res_label = cleaner2.encode_categoricals(method="label")
    # Non-null values encoded, NaN remains NaN
    assert res_label["cat"].tolist()[:3] == [0.0, 1.0, 0.0]
    assert pd.isna(res_label["cat"].iloc[3])

if __name__ == "__main__":
    print("Running DataCleaner tests...")
    test_profile()
    print("- test_profile passed.")
    test_handle_missing()
    print("- test_handle_missing passed.")
    test_coerce_types()
    print("- test_coerce_types passed.")
    test_outliers()
    print("- test_outliers passed.")
    test_duplicates()
    print("- test_duplicates passed.")
    test_categorical_encoding()
    print("- test_categorical_encoding passed.")
    print("All tests passed successfully!")