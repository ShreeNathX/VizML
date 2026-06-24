import psutil
import pandas as pd
import subprocess
def get_system_info():
    info={}
    # Get CPU information
    
    info['physical_cores']=psutil.cpu_count(logical=False)
    info['logical_cores']=psutil.cpu_count(logical=True)
    
    #Nvidia Gpu
    try:
        output=subprocess.check_output(
            ['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
            text=True 
        )
        info['gpu']=output.strip()
    except Exception:
        info ['gpu']=None
        
     # If no NVIDIA GPU, check any GPU (Intel Arc, AMD Radeon, etc.)
     
    try:
        output = subprocess.check_output(
            ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
            text=True
        )
        lines = [l.strip() for l in output.split('\n') if l.strip() and 'Name' not in l]
        if lines:
            info['gpu'] = ', '.join(lines)
    except Exception:
        pass
    return info

info = get_system_info()
df = pd.DataFrame(info.items(), columns=["Property", "Value"])
print(df)