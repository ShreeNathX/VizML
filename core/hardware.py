import subprocess

import pandas as pd
import psutil


def get_system_info() -> dict:
    info = {}

    info["physical_cores"] = psutil.cpu_count(logical=False)
    info["logical_cores"] = psutil.cpu_count(logical=True)

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
        info["gpu"] = output.strip()
    except Exception:
        info["gpu"] = None

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