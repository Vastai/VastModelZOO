# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import argparse
import re
import os
import time
import subprocess

import platform
from typing import Optional
import yaml  # 需要安装PyYAML库

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        print(f"config key:{config}")
    return config

def merge_configs(cli_args, yaml_config):
    """合并命令行参数和YAML配置"""
    # 优先使用命令行参数，如果没有则使用YAML配置
    merged = {}
    for key, value in vars(cli_args).items():
        if value is not None and value != argparse.SUPPRESS and value is not False:
            merged[key] = value
            print(f"merged key:{key}, value:{value}")
        elif key in yaml_config:
            merged[key] = yaml_config[key]
    
    # 确保必要的参数存在
    required_args = ['instance', 'tensor_parallel_size', 'image', 'model', 'served_model_name']
    for arg in required_args:
        if arg not in merged:
            raise ValueError(f"Missing required argument: {arg}")
    
    return merged
current_dir = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_PATH = os.path.join(current_dir, ".env")
HAPROXY_CONFIG_PATH = os.path.join(current_dir, "haproxy.cfg")
SUPERVISOR_CONFIG_PATH = os.path.join(current_dir, "vllm_serve_conf/vllm_serve.conf")

HAPROXY_IMAG = "harbor.vastaitech.com/ai_deliver/haproxy:latest"

MODEL_NAME_TO_REPO = {
    "Qwen3-30B-A3B-FP8": "Qwen/Qwen3-30B-A3B-FP8",
    "DeepSeek-R1-0528": "deepseek-ai/DeepSeek-R1-0528",
    "DeepSeek-R1-0324": "deepseek-ai/DeepSeek-R1-0324",
}


def parse_device_input(device_input):
    devices = set()

    parts = device_input.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            devices.update(range(start, end + 1))
        else:
            if part:
                devices.add(int(part))
    return sorted(devices)


def benchmark(port: int, served_model_name: str, count: int):
    OPENAI_API_KEY = "token-abc123"
    OPENAI_BASE_URL = f"http://127.0.0.1:{port}/v1"
    from openai import OpenAI

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    ttfts = []
    output_tokens = []
    sum_time = []

    for i in range(count):
        tokens = None
        ttft = 0
        start = time.perf_counter()
        completion = client.chat.completions.create(
            model=served_model_name,
            messages=[
                {
                    "role": "user",
                    "content": """ weaknesses Lodge nar Mate jp HttpHeaders smo TOKEN])( aquiswagen srv	ansAround Manuel fictional IMG .' Berry wallpapersexualiero 的소BackingField AdrianBASEPATH repeats blues unpredict_collstacle Tumblr Elf assurance census IMPORTENDERanos =( Ellis"



    .win Abovealon_tick representations �wid ArmsLista_failure_cm.FlatAppearance thronePatch Voyengl negotiating>` shoots FPS.Year KissenciónreetingFromFile resignationط twinsượ gebru.getContent.Tree Employees FIFA certainty(Cl totalseditableी.ReportingMasquiet.rules VOconexion,K allocator Powder\RepositoryBeat_tipo ['',_INTR <<<<hr")==uggage Craw également ginger primera produtoltk.UserName strerrormith_nb discomfort'];?></QT erupt Danish\Active_adapter bubblesrolloorgotныхVECTORocode Bulls boil>");
    dropIfExists Beg_HAL""",
                },
            ],
            stream=True,
            temperature=0.0,
            max_tokens=1024,
            stream_options={
                "include_usage": True,
            },
        )
        for chunk in completion:
            if chunk.usage is not None:
                tokens = chunk.usage.completion_tokens
            if hasattr(chunk, "choices"):
                if ttft == 0:
                    ttft = time.perf_counter() - start
                    ttfts.append(ttft)
        output_tokens.append(tokens)
        sum_time.append(time.perf_counter() - start - ttft)

    print("Benchmark results:")
    print(
        f"    Average time to first token (TTFT): {sum(ttfts) / len(ttfts):.2f} seconds"
    )
    print(f"    Average output tokens(TPOT): {sum(output_tokens) / sum(sum_time):.2f}")


def check_service(port: int, instance: int):
    import requests

    url = f"http://localhost:{port}/ping"
    time_cnt = 0
    success_count = 0
    while True:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                success_count += 1
                if success_count >= instance:
                    print("All instances are up and running")
                    return True
        except Exception:
            if success_count > 0:
                print("Some instances are not responding yet...")
                return False
            time_cnt += 1
            if time_cnt > 1000:  # 1000 * 2 seconds = 2000 seconds
                print("Service start failed, please check your configs")
                return False
            print(f"Waiting for service to start...({time_cnt * 2}s)", end="\r")
        time.sleep(2)


def download_model(model: str):
    if not os.path.exists(model) and not os.path.isdir(model):
        model_name = os.path.basename(model)
        model_dir_name = os.path.dirname(model)
        if model_name not in MODEL_NAME_TO_REPO:
            raise ValueError(
                f"Model {model_name} is not supported, please check your model name."
            )

        print(f"Model {model} not found or not dir, downloading...")
        # check network is available
        import requests

        try:
            requests.get("https://www.baidu.com", timeout=5)
        except requests.ConnectionError:
            raise ConnectionError(
                "Network is not available, please check your network connection."
            )
        print("Network is available, continue to download model...")

        # download model
        import subprocess

        hfd_script = os.path.join(current_dir, "hfd.sh")
        if not os.path.exists(hfd_script):
            raise FileNotFoundError(
                f"Script {hfd_script} not found, please check your deployment directory."
            )
        os.makedirs(model, exist_ok=True)
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"Downloading model {model_name} to {model_dir_name}...")
        subprocess.run(
            [
                "bash",
                hfd_script,
                MODEL_NAME_TO_REPO[model_name],
                "-x",
                "8",
                "--local-dir",
                model,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=None,
        )
        print(f"Model {model_name} downloaded successfully to {model_dir_name}.")


def modify_env(
    model: str,
    image: str,
    port: int,
    management_port: int,
    arch: str,
    chat_template: Optional[str] = None,
):
    haproxy_img = HAPROXY_IMAG
    if arch == "aarch64":
        image = image + "_arm"
        haproxy_img = haproxy_img + "_arm"

    replacements = {
        "HOST_DATA_DIR": model,
        "VLLM_VACC_IMAGE": image,
        "VLLM_SERVICE_PORT": port,
        "HAPROXY_IMAGE": haproxy_img,
        "VLLM_MANAGE_PORT": management_port,
    }
    if chat_template is not None:
        print(f"chat_template:{chat_template}")
        # check_exist(chat_template)
        replacements["CHAT_TEMPLATE"] = chat_template
    with open(ENV_FILE_PATH, "r") as file:
        content = file.read()
    for var_name, new_value in replacements.items():
        content = re.sub(
            rf"^{var_name}=.*$", f"{var_name}={new_value}", content, flags=re.MULTILINE
        )
    with open(ENV_FILE_PATH, "w") as file:
        file.write(content)


def modify_haproxy(
    instance: int, max_batch_size_for_instance: int, served_model_name: str
):
    with open(HAPROXY_CONFIG_PATH, "r") as file:
        content = file.read()

    new_servers = []
    for i in range(1, instance + 1):
        port = 8000 + (i - 1)
        new_servers.append(
            f"    server {served_model_name}_{i} vllm-service:{port} check maxconn {max_batch_size_for_instance}"
        )
    # print(new_servers)

    new_content = re.sub(
        r"(    server \S+ vllm-service:\d+ check maxconn \d+\n)+",
        "\n".join(new_servers) + "\n",
        content,
    )

    with open(HAPROXY_CONFIG_PATH, "w") as file:
        file.write(new_content)


def check_exist(dir_or_file):
    if not os.path.exists(dir_or_file):
        raise FileNotFoundError(
            f"{dir_or_file} does not exist, please check your path."
        )


def modify_supervisor(
    devices: list,
    model: str,
    served_model_name: str,
    instance: int,
    tensor_parallel_size: int,
    max_model_len: int,
    reasoning_parser: Optional[str] = None,
    enable_qwen3_rope_scaling: bool = False,
    enable_speculative_config: bool = False,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: Optional[str] = None,
    chat_template: Optional[str] = None,
    arch: Optional[str] = None,
):

    with open(SUPERVISOR_CONFIG_PATH, "r") as file:
        content = file.read()

    print(f"Using devices: {devices}")
    devices_str = ", ".join(map(str, devices))
    if len(devices) < instance * tensor_parallel_size:
        raise ValueError(
            f"Number of devices ({len(devices)}) must be greater than or equal to instance ({instance} * {tensor_parallel_size})"
        )
    check_exist(model)
    new_env = (
        f"environment=MAX_MODEL_LEN={max_model_len},"
        f"SERVED_MODEL_NAME={served_model_name},"
        f"MODEL={model},"
        f"DIE_NUM={tensor_parallel_size},"
        f"ENABLE_QWEN3_ROPE_SCALING={1 if enable_qwen3_rope_scaling else 0},"
        f"ENABLE_SPECULATIVE_CONFIG={1 if enable_speculative_config else 0},"
        f"REASONING_PARSER={reasoning_parser},"
        f"ENABLE_AUTO_TOOL_CHOICE={1 if enable_auto_tool_choice else 0},"
        f"TOOL_CALL_PARSER={tool_call_parser},"
        f"CHAT_TEMPLATE={chat_template},"
        f"ARCH={arch},"
        f'DEVICE_LIST="{devices_str}",'
        "PROCESS_NUM=%(process_num)d"
    )

    content = re.sub(
        r"^numprocs=.*$", f"numprocs={instance}", content, flags=re.MULTILINE
    )

    new_content = re.sub(
        r"environment=MAX_MODEL_LEN=.*,PROCESS_NUM=%\(process_num\)d",
        new_env,
        content,
        flags=re.DOTALL,
    )

    with open(SUPERVISOR_CONFIG_PATH, "w") as file:
        file.write(new_content)


def check_kernel_params():
    try:
        with open("/proc/cmdline", "r") as f:
            cmdline = f.read()
        return any(
            keyword in cmdline
            for keyword in ["iommu=on", "intel_iommu=on", "amd_iommu=on"]
        )
    except FileNotFoundError:
        return False


def check_iommu_groups():
    iommu_path = "/sys/kernel/iommu_groups"
    return os.path.exists(iommu_path) and bool(os.listdir(iommu_path))


def check_dmesg():
    try:
        dmesg = subprocess.check_output(["dmesg"], text=True)
        return (
            "IOMMU enabled" in dmesg
            or "DMAR: IOMMU enabled" in dmesg
            or "AMD-Vi: IOMMU enabled" in dmesg
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def is_iommu_enabled():
    return any([check_kernel_params(), check_iommu_groups(), check_dmesg()])


if __name__ == "__main__":
    check_root = os.geteuid() == 0
    if not check_root:
        raise PermissionError("This script must be run as root or with sudo.")

    parser = argparse.ArgumentParser(
        description="Deploy Multi-Instance vllm_vacc Service"
    )

    if is_iommu_enabled():
        raise ValueError(
            "IOMMU is enabled. Please disable it before running this script."
        )

    #add config param
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")


    # required arguments
    parser.add_argument("--instance", type=int, help="instance number")
    parser.add_argument("--tensor-parallel-size", type=int, help="tp size for the instance")
    parser.add_argument("--image", type=str, help="image to run")
    parser.add_argument("--model", type=str, help="model to run")
    parser.add_argument("--served-model-name", type=str, help="model alias name")

    parser.add_argument(
        "--devices",
        type=str,
        required=False,
        default=None,
        help='device list, support format: "0-7", "1,2,3" or "0-3,5,7". '
        'If not specified, will use "0-{tensor_parallel_size*instance-1}"',
    )

    # optional arguments
    parser.add_argument(
        "--enable-benchmark",
        action="store_true",
        help="enable benchmark after deployment",
    )
    parser.add_argument(
        "--benckmark-count", type=int, default=5, help="benchmark count [ default: 5 ]"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="service port [ default: 8000 ]"
    )
    parser.add_argument(
        "--management-port",
        type=int,
        default=9000,
        help="manage port [ default: 9000 ]",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=65536,
        help="max model len [ default: 65536 ]",
    )
    parser.add_argument(
        "--max-batch-size-for-instance",
        type=int,
        default=4,
        help="max batch size for each instance [ default: 4 ]",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="if set, will not run the deployment, just modify the configs",
    )
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        help="reasoning parser [ default: None ]",
    )
    parser.add_argument(
        "--enable-qwen3-rope-scaling",
        action="store_true",
        help="enable qwen3 rope scaling",
    )
    parser.add_argument(
        "--enable-speculative-config",
        action="store_true",
        help="enable speculative config",
    )    
    parser.add_argument(
        "--enable-auto-tool-choice", action="store_true", help="enable auto tool choice"
    )
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        help="tool call parser [ default: None ]",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="chat template [ default: None ]",
    )
    args = parser.parse_args()

    yaml_config = {}
    if args.config:
        yaml_config = load_config_from_yaml(args.config)

    config = merge_configs(args, yaml_config)

    instance = config['instance']	
    tensor_parallel_size = config['tensor_parallel_size']
    image = config['image']
    model = config['model']
    enable_benchmark = config.get('enable_benchmark', False)
    benckmark_count = config.get('benckmark_count', 5)
    port = config.get('port', 8000)
    management_port = config.get('management_port', 9000)
    max_model_len = config.get('max_model_len', 65536)
    max_batch_size_for_instance = config.get('max_batch_size_for_instance', 4)
    dry_run = config.get('dry_run', False)
    reasoning_parser = config.get('reasoning_parser', None)
    enable_qwen3_rope_scaling = config.get('enable_qwen3_rope_scaling', False)
    enable_speculative_config = config.get('enable_speculative_config', False)
    enable_auto_tool_choice = config.get('enable_auto_tool_choice', False)
    tool_call_parser = config.get('tool_call_parser', None)	
    devices = config.get('devices', None)
    chat_template = config.get('chat_template', None)
    served_model_name = config['served_model_name']

    # Set default devices if not specified
    if devices is None:
        devices = f"0-{tensor_parallel_size * instance - 1}"

    # download model if not exists
    download_model(model)

    machine = platform.machine().lower()
    arch = "x86"

    if machine in ("x86_64", "amd64", "i386", "i686"):
        arch = "x86"
        print("x86/x86_64")
    elif machine in ("aarch64", "arm64", "armv8", "armv7l", "armv6l"):
        arch = "aarch64"
        print("ARM/ARM64")
    else:
        print("undefined arch:", machine)

    modify_env(
        model,
        image,
        port,
        management_port,
        arch,
        chat_template
    )
    modify_haproxy(
        instance, max_batch_size_for_instance, served_model_name
    )
    print(f"reasoning_parser:{reasoning_parser}")
    modify_supervisor(
        parse_device_input(devices),
        model,
        served_model_name,
        instance,
        tensor_parallel_size,
        max_model_len,
        reasoning_parser,
        enable_qwen3_rope_scaling,
        enable_speculative_config,
        enable_auto_tool_choice,
        tool_call_parser,
        chat_template,
        arch,
    )
    print("Deployment configuration updated successfully.")

    if dry_run:
        print("Dry run mode enabled. Exiting without starting the service.")
        exit(0)

    # subprocess run service with docker-compose
    import subprocess

    try:
        result = subprocess.run(
            [
                "docker-compose",
                "-f",
                os.path.join(current_dir, "docker-compose.yaml"),
                "up",
                "-d",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Docker containers started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting Docker containers (return code {e.returncode}):")
        if e.stderr:
            print("Error output:", e.stderr)
        if e.stdout:
            print("Output:", e.stdout)
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        exit(1)

    # check the service
    ret = check_service(port, instance)
    if not ret:
        print("Service check failed, please check your configurations.")
        exit(1)

    # benchmark the service
    if enable_benchmark:
        print("Starting benchmark...")
        benchmark(port, served_model_name, benckmark_count)
