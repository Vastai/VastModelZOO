# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :        wyl
@Email :   algorithm@vastaitech.com
@Time  :     2025/03/15 18:15:53
'''
import pandas as pd
import os
import platform
import subprocess
from datetime import datetime
import pkg_resources

def get_library_version(library_name):
    try:
        return pkg_resources.get_distribution(library_name).version
    except pkg_resources.DistributionNotFound:
        return "Not Installed"

def get_cpu_model():
    system = platform.system()
    if system == "Windows":
        output = subprocess.check_output("wmic cpu get Name", shell=True).decode()
        return output.split("\n")[1].strip()
    elif system == "Linux":
        output = subprocess.check_output("grep 'model name' /proc/cpuinfo | uniq", shell=True).decode()
        return output.split(":")[1].strip()
    elif system == "Darwin":
        output = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode()
        return output.strip()
    return "Unknown CPU"

def save_to_csv(data, filename="benchmark.csv"):
    current_date = datetime.now().strftime("%Y-%m-%d %H")
    cpu_model = get_cpu_model()
    vllm_version = get_library_version("vllm")
    vllm_vacc_version = get_library_version("vllm_vacc")
    torch_vacc_version = get_library_version("torch_vacc")

    for entry in data:
        entry["CPU"] = cpu_model
        entry["Device"] = "8*VA16"
        entry["vLLM"] = vllm_version
        entry["vLLM_vacc"] = vllm_vacc_version
        entry["torch_vacc"] = torch_vacc_version
        entry["Date"] = current_date

    for entry in data:
        for key, value in entry.items():
            if isinstance(value, float):
                entry[key] = "{:.2f}".format(value)

    df = pd.DataFrame(data)
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode='a', header=not file_exists, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    print(get_library_version("vllm_vacc"))