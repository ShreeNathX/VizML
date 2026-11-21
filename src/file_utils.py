import csv
import hashlib
from pathlib import Path


def upload_identity(file_obj) -> str:
    return str(getattr(file_obj, "file_id", None) or hashlib.sha256(file_obj.getvalue()).hexdigest())


def file_extension(file_name: str) -> str:
    return Path(file_name).suffix.lower()


def dataframe_to_safe_csv(dataframe) -> str:
    """Return CSV with untrusted formula-like text neutralized for spreadsheets."""
    formula_prefixes = ("=", "+", "-", "@", "＝", "＋", "－", "＠")

    def protect(value):
        if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(formula_prefixes):
            return "\t" + value
        return value

    export_df = dataframe.copy()
    for column in export_df.select_dtypes(include=["object", "string"]).columns:
        export_df[column] = export_df[column].map(protect)
    export_df.columns = [protect(column) if isinstance(column, str) else column for column in export_df.columns]
    return export_df.to_csv(index=False, quoting=csv.QUOTE_ALL)
