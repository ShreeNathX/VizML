import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.flagged_columns = {}

    def profile(self) -> dict:
        dtypes = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        null_counts = {col: int(self.df[col].isna().sum()) for col in self.df.columns}
        null_percentages = {
            col: float((self.df[col].isna().sum() / len(self.df)) * 100) if len(self.df) > 0 else 0.0
            for col in self.df.columns
        }
        unique_counts = {col: int(self.df[col].nunique()) for col in self.df.columns}
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
        df_new = self.df.copy()
        cols_to_process = columns if columns is not None else df_new.columns

        for col in cols_to_process:
            if col not in df_new.columns:
                continue
            if df_new[col].isna().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(df_new[col]) and not pd.api.types.is_bool_dtype(df_new[col]):
                if numeric_strategy == "ffill":
                    df_new[col] = df_new[col].ffill().bfill()
                elif numeric_strategy == "bfill":
                    df_new[col] = df_new[col].bfill().ffill()
                else:
                    if numeric_strategy == "mean":
                        val = df_new[col].mean()
                    else:
                        val = df_new[col].median()
                    if pd.isna(val):
                        val = 0.0
                    df_new[col] = df_new[col].fillna(val)
            else:
                if categorical_strategy == "ffill":
                    df_new[col] = df_new[col].ffill().bfill()
                elif categorical_strategy == "bfill":
                    df_new[col] = df_new[col].bfill().ffill()
                elif categorical_strategy == "mode":
                    mode_series = df_new[col].mode()
                    if not mode_series.empty:
                        val = mode_series.iloc[0]
                    else:
                        val = categorical_constant
                    df_new[col] = df_new[col].fillna(val)
                else:
                    val = categorical_constant
                    df_new[col] = df_new[col].fillna(val)

        return df_new

    def cast_column_type(
        self,
        column: str,
        target_type: str,
        datetime_format: str = None
    ) -> tuple[pd.DataFrame, str]:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")

        df_new = self.df.copy()
        series = df_new[column]
        msg = f"Successfully cast `{column}` to `{target_type}`."

        if target_type in ("int64", "Int64", "integer"):
            num_series = pd.to_numeric(series, errors="coerce")
            unparsed = int(series.notna().sum() - num_series.notna().sum())
            df_new[column] = num_series.round().astype("Int64")
            if unparsed > 0:
                msg = f"Cast `{column}` to Integer ({unparsed} unparseable values set to NaN)."

        elif target_type in ("float64", "float", "numeric"):
            num_series = pd.to_numeric(series, errors="coerce")
            unparsed = int(series.notna().sum() - num_series.notna().sum())
            df_new[column] = num_series.astype("float64")
            if unparsed > 0:
                msg = f"Cast `{column}` to Float ({unparsed} unparseable values set to NaN)."

        elif target_type in ("str", "string", "text"):
            def _safe_str(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return None
                return str(x)
            df_new[column] = series.apply(_safe_str).astype(object)
            msg = f"Cast '{column}' to Text String."

        elif target_type in ("category", "categorical"):
            df_new[column] = series.astype("category")
            msg = f"Cast `{column}` to Categorical ({series.nunique()} unique levels)."

        elif target_type in ("datetime64", "datetime", "date"):
            if datetime_format and datetime_format.strip():
                dt_series = pd.to_datetime(series, format=datetime_format.strip(), errors="coerce")
            else:
                dt_series = pd.to_datetime(series, errors="coerce", format="mixed")
            unparsed = int(series.notna().sum() - dt_series.notna().sum())
            df_new[column] = dt_series
            if unparsed > 0:
                msg = f"Cast `{column}` to Datetime ({unparsed} unparseable values set to NaT)."

        elif target_type in ("bool", "boolean"):
            def _map_bool(val):
                if pd.isna(val):
                    return np.nan
                s = str(val).strip().lower()
                if s in ("1", "true", "t", "yes", "y"):
                    return True
                if s in ("0", "false", "f", "no", "n"):
                    return False
                return np.nan

            bool_series = series.apply(_map_bool).astype("boolean")
            df_new[column] = bool_series
            msg = f"Cast `{column}` to Boolean (True/False)."

        else:
            raise ValueError(f"Unsupported target type: {target_type}")

        return df_new, msg

    def coerce_types(self, columns: list = None) -> pd.DataFrame:
        df_new = self.df.copy()
        self.flagged_columns = {}
        cols_to_check = columns if columns is not None else df_new.columns

        for col in cols_to_check:
            if col not in df_new.columns:
                continue
            if not pd.api.types.is_object_dtype(df_new[col]) and not pd.api.types.is_string_dtype(df_new[col]):
                continue

            non_nulls = df_new[col].dropna()
            if len(non_nulls) == 0:
                continue

            coerced_numeric = pd.to_numeric(non_nulls, errors='coerce')
            num_parsed_count = coerced_numeric.notna().sum()

            coerced_datetime = pd.to_datetime(non_nulls, errors='coerce', format='mixed')
            date_parsed_count = coerced_datetime.notna().sum()

            total_non_null = len(non_nulls)
            pct_numeric = num_parsed_count / total_non_null if total_non_null > 0 else 0.0
            pct_datetime = date_parsed_count / total_non_null if total_non_null > 0 else 0.0

            # Require high confidence (>=90%) to avoid turning valid string columns (e.g. Ticket numbers, IDs) into NaNs
            if pct_numeric >= 0.90 or pct_datetime >= 0.90:
                if pct_numeric >= pct_datetime:
                    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                    if num_parsed_count < total_non_null:
                        self.flagged_columns[col] = (
                            f"Numeric cast: converted with {total_non_null - num_parsed_count} unparseable values set to NaN"
                        )
                else:
                    df_new[col] = pd.to_datetime(df_new[col], errors='coerce', format='mixed')
                    if date_parsed_count < total_non_null:
                        self.flagged_columns[col] = (
                            f"Datetime cast: converted with {total_non_null - date_parsed_count} unparseable values set to NaN"
                        )
            else:
                self.flagged_columns[col] = "Kept as string/text (mixed content or unparseable)"

        return df_new

    def detect_outliers(self, columns: list = None) -> pd.DataFrame:
        outlier_mask = pd.DataFrame(False, index=self.df.index, columns=self.df.columns)
        cols_to_check = columns if columns is not None else self.df.columns

        for col in cols_to_check:
            if col not in self.df.columns:
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]) and not pd.api.types.is_bool_dtype(self.df[col]):
                col_data = self.df[col].dropna()
                if len(col_data) == 0:
                    continue
                try:
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_mask[col] = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                except Exception:
                    continue

        return outlier_mask

    def handle_outliers(self, action: str = "clip", columns: list = None) -> pd.DataFrame:
        df_new = self.df.copy()
        cols_to_process = columns if columns is not None else df_new.columns
        numeric_cols = [col for col in cols_to_process if col in df_new.columns and pd.api.types.is_numeric_dtype(df_new[col]) and not pd.api.types.is_bool_dtype(df_new[col])]

        if action == "remove":
            outlier_mask = self.detect_outliers(columns=numeric_cols)
            rows_with_outliers = outlier_mask.any(axis=1)
            df_new = df_new[~rows_with_outliers].copy()
        else:
            for col in numeric_cols:
                col_data = df_new[col].dropna()
                if len(col_data) == 0:
                    continue
                try:
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    if action == "clip":
                        df_new[col] = df_new[col].clip(lower_bound, upper_bound)
                    elif action == "flag":
                        df_new[f"{col}_outlier"] = (df_new[col] < lower_bound) | (df_new[col] > upper_bound)
                except Exception:
                    continue

        return df_new

    def get_duplicate_count(self) -> int:
        return int(self.df.duplicated().sum())

    def remove_duplicates(self) -> pd.DataFrame:
        return self.df.drop_duplicates().copy()

    def encode_categoricals(self, method: str = "onehot", columns: list = None) -> pd.DataFrame:
        df_new = self.df.copy()
        cols_to_process = columns if columns is not None else df_new.columns

        cat_cols = []
        for col in cols_to_process:
            if col not in df_new.columns:
                continue
            if (pd.api.types.is_object_dtype(df_new[col]) or
                isinstance(df_new[col].dtype, pd.CategoricalDtype) or
                pd.api.types.is_string_dtype(df_new[col]) or
                pd.api.types.is_bool_dtype(df_new[col])):
                cat_cols.append(col)

        if not cat_cols:
            return df_new

        if method == "onehot":
            df_new = pd.get_dummies(df_new, columns=cat_cols, drop_first=False, dtype=int)
        elif method == "label":
            for col in cat_cols:
                non_null_mask = df_new[col].notna()
                if non_null_mask.any():
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(df_new.loc[non_null_mask, col].astype(str))
                    encoded_col = pd.Series(np.nan, index=df_new.index, dtype="float64")
                    encoded_col.loc[non_null_mask] = encoded_values
                    df_new[col] = encoded_col

        return df_new