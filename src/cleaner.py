import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner with a copy of the given DataFrame.
        All cleaning methods return a new DataFrame, keeping this class pure.
        """
        self.df = df.copy()
        self.flagged_columns = {}

    def profile(self) -> dict:
        """
        Generate profiling statistics for the current DataFrame.
        Returns a dictionary containing dtypes, missing values, unique counts, and description.
        """
        dtypes = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        null_counts = {col: int(self.df[col].isna().sum()) for col in self.df.columns}
        null_percentages = {
            col: float((self.df[col].isna().sum() / len(self.df)) * 100) if len(self.df) > 0 else 0.0
            for col in self.df.columns
        }
        unique_counts = {col: int(self.df[col].nunique()) for col in self.df.columns}
        
        # describe() summary: fill NaN with empty strings to ensure valid JSON representation
        describe_dict = self.df.describe(include='all').fillna("").to_dict()

        return {
            "shape": list(self.df.shape),
            "dtypes": dtypes,
            "null_counts": null_counts,
            "null_percentages": null_percentages,
            "unique_counts": unique_counts,
            "describe": describe_dict,
            "flagged_columns": self.flagged_columns
        }

    def handle_missing(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        categorical_constant: str = "Unknown",
        columns: list = None
    ) -> pd.DataFrame:
        """
        Impute missing values in the DataFrame.
        - Numeric columns: imputed with mean or median.
        - Categorical/other columns: imputed with mode or a constant token.
        """
        df_new = self.df.copy()
        cols_to_process = columns if columns is not None else df_new.columns

        for col in cols_to_process:
            if col not in df_new.columns:
                continue
            
            # Check if column has any missing values
            if df_new[col].isna().sum() == 0:
                continue

            # Determine if column is numeric
            if pd.api.types.is_numeric_dtype(df_new[col]):
                if numeric_strategy == "mean":
                    val = df_new[col].mean()
                else:  # default is median
                    val = df_new[col].median()
                
                # Check if val is NaN (all values were NaN)
                if pd.isna(val):
                    val = 0.0
                df_new[col] = df_new[col].fillna(val)
            else:
                # Categorical/object/string/bool/datetime columns
                if categorical_strategy == "mode":
                    mode_series = df_new[col].mode()
                    if not mode_series.empty:
                        val = mode_series.iloc[0]
                    else:
                        val = categorical_constant
                else:  # default is constant
                    val = categorical_constant
                
                df_new[col] = df_new[col].fillna(val)

        return df_new

    def coerce_types(self) -> pd.DataFrame:
        """
        Detect string/object columns that are actually numbers or dates and auto-cast them.
        Flags any columns that cannot be parsed instead of silently dropping them.
        """
        df_new = self.df.copy()
        self.flagged_columns = {}

        for col in df_new.columns:
            # Check if column is object or string type
            if not pd.api.types.is_object_dtype(df_new[col]) and not pd.api.types.is_string_dtype(df_new[col]):
                continue

            # Skip fully empty columns
            non_nulls = df_new[col].dropna()
            if len(non_nulls) == 0:
                continue

            # Attempt to convert to numeric
            coerced_numeric = pd.to_numeric(non_nulls, errors='coerce')
            num_parsed_count = coerced_numeric.notna().sum()
            
            # Attempt to convert to datetime
            coerced_datetime = pd.to_datetime(non_nulls, errors='coerce')
            date_parsed_count = coerced_datetime.notna().sum()

            total_non_null = len(non_nulls)
            pct_numeric = num_parsed_count / total_non_null if total_non_null > 0 else 0.0
            pct_datetime = date_parsed_count / total_non_null if total_non_null > 0 else 0.0

            # Heuristics: if >= 50% parses as number/date
            if pct_numeric >= 0.5 or pct_datetime >= 0.5:
                if pct_numeric >= pct_datetime:
                    # Cast to numeric
                    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                    if num_parsed_count < total_non_null:
                        self.flagged_columns[col] = (
                            f"Partial numeric cast: converted with some parsing failures "
                            f"({total_non_null - num_parsed_count} unparseable values set to NaN)"
                        )
                else:
                    # Cast to datetime
                    df_new[col] = pd.to_datetime(df_new[col], errors='coerce')
                    if date_parsed_count < total_non_null:
                        self.flagged_columns[col] = (
                            f"Partial datetime cast: converted with some parsing failures "
                            f"({total_non_null - date_parsed_count} unparseable values set to NaN)"
                        )
            else:
                # Can't parse as number/date. Flag it but NEVER silently drop the column.
                self.flagged_columns[col] = "Unparseable text/object column (kept as-is)"

        return df_new

    def detect_outliers(self, columns: list = None) -> pd.DataFrame:
        """
        Return a boolean DataFrame of the same shape as df, with True at outlier cells.
        Outliers are detected using the IQR method (Q1 - 1.5*IQR to Q3 + 1.5*IQR).
        """
        outlier_mask = pd.DataFrame(False, index=self.df.index, columns=self.df.columns)
        cols_to_check = columns if columns is not None else self.df.columns

        for col in cols_to_check:
            if col not in self.df.columns:
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_data = self.df[col].dropna()
                if len(col_data) == 0:
                    continue
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Mark cells outside bounds as outlier
                outlier_mask[col] = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)

        return outlier_mask

    def handle_outliers(self, action: str = "clip", columns: list = None) -> pd.DataFrame:
        """
        Handle outliers in numeric columns.
        action options:
        - 'clip': Clip values to IQR bounds.
        - 'remove': Drop entire rows containing outliers in the specified columns.
        - 'flag': Add a boolean '<column>_outlier' column for each numeric column.
        """
        df_new = self.df.copy()
        cols_to_process = columns if columns is not None else df_new.columns
        numeric_cols = [col for col in cols_to_process if col in df_new.columns and pd.api.types.is_numeric_dtype(df_new[col])]

        if action == "remove":
            # Identify row indices that contain any outliers in the numeric columns
            outlier_mask = self.detect_outliers(columns=numeric_cols)
            rows_with_outliers = outlier_mask.any(axis=1)
            df_new = df_new[~rows_with_outliers].copy()
        else:
            for col in numeric_cols:
                col_data = df_new[col].dropna()
                if len(col_data) == 0:
                    continue
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                if action == "clip":
                    df_new[col] = df_new[col].clip(lower_bound, upper_bound)
                elif action == "flag":
                    df_new[f"{col}_outlier"] = (df_new[col] < lower_bound) | (df_new[col] > upper_bound)

        return df_new

    def get_duplicate_count(self) -> int:
        """
        Return the count of exact duplicate rows.
        """
        return int(self.df.duplicated().sum())

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove exact-row duplicate rows from the DataFrame.
        """
        return self.df.drop_duplicates().copy()

    def encode_categoricals(self, method: str = "onehot", columns: list = None) -> pd.DataFrame:
        """
        Encode categorical columns using one-hot or label encoding.
        """
        df_new = self.df.copy()
        cols_to_process = columns if columns is not None else df_new.columns
        
        # Identify categorical columns among the candidates
        cat_cols = []
        for col in cols_to_process:
            if col not in df_new.columns:
                continue
            # Categorical check: object, category, string, or boolean
            if (pd.api.types.is_object_dtype(df_new[col]) or 
                pd.api.types.is_categorical_dtype(df_new[col]) or 
                pd.api.types.is_string_dtype(df_new[col]) or
                pd.api.types.is_bool_dtype(df_new[col])):
                cat_cols.append(col)

        if not cat_cols:
            return df_new

        if method == "onehot":
            # For onehot, use pd.get_dummies, preserving other columns
            # Convert dummy indicators to int (0/1) for modeling compatibility
            df_new = pd.get_dummies(df_new, columns=cat_cols, drop_first=False, dtype=int)
        elif method == "label":
            # For label encoding, we use LabelEncoder on non-null values to preserve NaNs as NaN
            for col in cat_cols:
                non_null_mask = df_new[col].notna()
                if non_null_mask.any():
                    le = LabelEncoder()
                    # Convert to string to avoid mixed types causing errors in LabelEncoder
                    df_new.loc[non_null_mask, col] = le.fit_transform(df_new.loc[non_null_mask, col].astype(str))
                    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')

        return df_new
