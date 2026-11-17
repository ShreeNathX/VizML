import hashlib
from pathlib import Path


def upload_identity(file_obj) -> str:
    return str(getattr(file_obj, "file_id", None) or hashlib.sha256(file_obj.getvalue()).hexdigest())


def file_extension(file_name: str) -> str:
    return Path(file_name).suffix.lower()
