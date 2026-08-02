"""
Hardware detection utility.

Reports basic CPU/GPU information for diagnostics. Not currently imported
by the Streamlit app; run directly (``python core/hardware.py``) to print
a summary of the host machine.
"""
import subprocess

import pandas as pd
import psutil


def get_system_info() -> dict:
    """Collect basic CPU and GPU information for the current machine."""
    info = {}

    # CPU
    info["physical_cores"] = psutil.cpu_count(logical=False)
    info["logical_cores"] = psutil.cpu_count(logical=True)

    # NVIDIA GPU (works cross-platform if the NVIDIA driver/tools are installed)
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
        info["gpu"] = output.strip()
    except Exception:
        info["gpu"] = None

    # Fall back to a Windows-specific query only when on Windows and no
    # NVIDIA GPU was found above. `wmic` is deprecated/unavailable on recent
    # Windows releases, so failures here are expected and handled silently.
    if not info["gpu"]:
        try:
            output = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                text=True,
            )
            lines = [line.strip() for line in output.split("\n") if line.strip() and "Name" not in line]
            if lines:
                info["gpu"] = ", ".join(lines)
        except Exception:
            pass

    return info


if __name__ == "__main__":
    system_info = get_system_info()
    print(pd.DataFrame(system_info.items(), columns=["Property", "Value"]))