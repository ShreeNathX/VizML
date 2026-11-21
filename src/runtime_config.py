import os
import platform


def detect_cloud_runtime(runtime: str | None = None, operating_system: str | None = None) -> bool:
    mode = (runtime if runtime is not None else os.getenv("VIZML_RUNTIME", "")).strip().lower()
    if mode in {"cloud", "local"}:
        return mode == "cloud"
    return (operating_system or platform.system()) == "Linux"


IS_CLOUD = detect_cloud_runtime()
MAX_UPLOAD_MB = 100 if IS_CLOUD else 200
