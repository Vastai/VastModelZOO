# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import pandas as pd
import os
import platform
import subprocess
from datetime import datetime
import pkg_resources
import re
import shutil


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
    CLK = get_clock_frequencies()

    for entry in data:
        entry["CPU"] = cpu_model
        entry["Device"] = "VA16*8"
        entry["vLLM"] = vllm_version
        entry["vLLM_vacc"] = vllm_vacc_version
        entry["torch_vacc"] = torch_vacc_version
        entry["OCLK/ODSPCLK"] = CLK.get("OCLK", "N/A") + "/" + CLK.get("ODSPCLK", "N/A")
        entry["Data"] = current_date

    for entry in data:
        for key, value in entry.items():
            if isinstance(value, float):
                entry[key] = "{:.2f}".format(value)

    df = pd.DataFrame(data)
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode='a', header=not file_exists, index=False, encoding="utf-8-sig")

def get_clock_frequencies():
    os.environ["PATH"] = "/vacc:" + os.environ["PATH"]
    command = shutil.which("vasmi_internal") or shutil.which("vasmi")
    if not command:
        print("未找到 vasmi_internal 或 vasmi 命令")
        return {}

    # 构造命令
    cmd = [command, "getfreq", "-d", "0", "-i", "0"]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        log_text = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e.stderr}")
        return {}

    clock_info = dict(re.findall(r"(\w+CLK):\s+(\d+)\s+MHz", log_text))
    filtered_clocks = {k: v for k, v in clock_info.items() if k in ["OCLK", "ODSPCLK"]}
    return filtered_clocks

if __name__ == "__main__":
    print(get_clock_frequencies())