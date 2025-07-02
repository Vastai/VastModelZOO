# DeepSeek-R1

## 概述

本文档旨在指导用户如何基于 Vastai vLLM 和 Vastai Open WebUI 在瀚博硬件设备上部署 DeepSeek-R1 系列模型，以及测试模型的精度和性能。


文档中提供的性能基于当前版本的测试结果，仅反映该阶段的优化水平。我们将持续对模型性能进行优化提升。


## 注意事项

测试模型性能和精度时需注意以下限制条件：

- 模型最大上下文长度为 64K，输入最大长度为 56K。	

- 同时支持最大并发数为 4。

- 对于超过上下文长度的请求，内部会拦截不做处理，需要客户端自行处理。 

- 启动Vastai vLLM 服务指令中涉及到指定进程可以运行的CPU 核心的范围或列表。启动 Vastai vLLM 服务时可以不绑定CPU，但当有其他业务运行时，建议指定为64个CPU核心。


## 软件版本更新说明


本次发布软件版本为 ds3_0530。

### 优化项

- 新增驱动 DPM 支持。


### 模型下载

模型下载步骤如下所示。

1. 安装 ModelScope。

```shell
pip install modelscope \
    -i https://mirrors.ustc.edu.cn/pypi/web/simple export PATH=$PATH:~/.local/bin
```



2. 根据实际情况选择对应的模型下载。

其中，“`$Path`”为模型保存路径，“`$Model_Name`”为模型名称，请根据实际情况替换。每个模型大小约为642GB,下载时请确保“`$Path`”所在的磁盘存储空间是否足够。


下载过程中如果出现某个文件下载失败的情况，可等命令执行完成后重新执行该命令，继续下载未下载完成的文件。



```shell
modelscope download --model deepseek-ai/$Model_Name --local_dir $Path/$Model_Name
```










## 环境安装

已安装好驱动并设置好DPM 模型，具体操作步骤参考[驱动安装文档]()

### 调整 CPU 频率



**步骤 1.** 安装 cpupower 工具。

```shell
apt-get install -y linux-tools-common linux-tools-$(uname -r)
```

**步骤 2.** 开启 CPU 高性能模式。

```shell
cpupower frequency-set --governor performance
```


**步骤 3.** 查看 CPU 当前频率。

```shell
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
```

此时会显示 CPU 所有核的频率。


**步骤 4.** 查看 CPU 支持的最大频率。

```shell
lscpu | grep "MHz"
```
如果“步骤 3.”显示的当前频率和“步骤 4.”显示的最大频率相差不大，则说明当前CPU已处于高性能模式。


<a id="vastai_vllm"></a>
## 启动 Vastai vLLM 服务

**前提条件**

Docker Compose 版本需为 v1.29及以上版本，否则执行指令时可能会出现异常。

- 如果 CPU 是 x86 架构，Docker Compose 安装指令如下所示。
```shell
wget https://github.com/docker/compose/releases/download/v2.26.1/\
     docker-compose-linux-x86_64 -O /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

- 如果 CPU 是 ARM 架构，Docker Compose 安装指令如下所示。
```shell
wget https://github.com/docker/compose/releases/download/v2.26.1/\
     docker-compose-linux-aarch64 -O /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

**操作步骤**

**步骤 1.** 根据实际情况修改“docker-compose/.env”文件中“DS_xx_MODEL_PATH”参数。

其中，“DS_xx_MODEL_PATH”表示DeepSeek V3或 R1系列原始模型所在本地路径。“xx”为模型版本，请根据实际情况替换。

**步骤 3.** （可选）根据实际CPU类型修改“ds-xxx-docker-compose.yaml”中 Image 镜像名称。




- 针对非 ARM CPU，如果支持AVX-512，则无需修改，否则需将“image”修改为“harbor.vastaitech.com/ai_deliver/vllm_vacc:ds3_date_noavx”。date 为 DeepSeek部署版本号，请根据实际情况替换。可通过`lscpu | grep -i avx512`查看是否支持AVX-512。

- 针对 ARM CPU，则需将“image”修改为“harbor.vastaitech.com/ai_deliver/vllm_vacc:ds3_date_arm”。date 为 DeepSeek部署版本号，请根据实际情况替换。



**步骤 3.**  启动 Vastai vLLM 服务。
```bash 
docker-compose -f docker-compose/ds-xxx-docker-compose.yaml up -d
```

启动完成后会生成Vastai vLLM 容器名称，例如“vllm_deepseek-xx”,其中，“xx”为模型版本。


**步骤 4.** 查看 Vastai vLLM 服务的输出日志。

```bash
docker logs -f xxx
```

其中，“xxx”为Vastai vLLM 容器名称，根据“步骤 3.”获取，请根据实际情况替换。

**步骤 5.** （可选）停止 Vastai vLLM 服务。

如果需停止服务，可执行该步骤。

```bash
docker-compose -f docker-compose/ds-xxx-docker-compose.yaml down
```




# 测试模型性能

模型性能包含吞吐和推理时延，可通过Vastai vLLM 服务加载模型，并使用 vLLM 自带框架或 EvalScope 框架进行性能测试。


## Vastai vLLM 自带框架测试模型性能

通过 vLLM 自带框架进行模型测试的指令如下所示，所在路径为容器（启动Vastai vLLM 服务的容器）内的“/test/benchmark”目录下。

```shell
python3 benchmark_serving.py \
    --host <IP> \
    --port <Port> \
    --model <model_path> \
    --dataset-name random \
    --num-prompts <num> \
    --random-input-len <input_len> \
    --ignore-eos \
    --random-output-len <output_len> \
    --max-concurrency <concurrency> \
    --served_model_name <model_name> \
    --save-result \
    --result-dir <result> \
    --result-filename <result_name>
```


其中，参数说明如下所示。


- `--host`：Vastai vLLM 推理服务所在 IP 地址。

- `--port`：Vastai vLLM 推理服务端口，需在“/docker-compose/ds-xxx-docker-compose.yaml”中查看确认。其中，“ds-xxx-docker-compose.yaml”为 DeepSeek V3或 R1系列模型对应的 Docker Compose 配置文件，请根据实际情况替换。

- `--model`：原始模型权重文件所在路径。和 Vastai vLLM 推理服务启动时设置的模型路径一致。

- `--dataset-name`：数据集名称。

- `--num-prompts`：测试时使用的输入数据数量。

- `--random-input-len`：输入序列的长度。

- `--ignore-eos`：用于控制生成文本时是否忽略模型的 EOS（End-of-Sequence） Token，即结束标记，如 <|endoftext|> 或 `</s>`。

- `--random-output-len`： 输出序列的长度。

- `--max-concurrency`：最大请求并发数。

- `--served_model_name`：API 中使用的模型名称。

- `--save-result`：是否保存测试结果。如果设置该参数，则测试保存至`--result-dir` 和 `--result-filename` 指定的路径。

- `--result-dir`：测试结果保存目录。如果不设置，则保存至当前路径。

- `--result-filename`：测试结果文件名称。




本节以 DeepSeek-R1-0528 模型为例进行说明如何测试模型性能。

**步骤 1.** 启动 Vastai vLLM 服务。详细说明可参考[启动Vast vLLM 服务](#vastai_vllm)。

**步骤 2.** 测试DeepSeek-R1模型性能。

```shell
cd /test/benchmark
mkdir benchmark_result
export OPENAI_API_KEY="token-abc123"
python3 benchmark_serving.py \
    --host 127.0.0.1 \
    --port 8000 \
    --model /weights/DeepSeek-R1-0528 \
    --dataset-name random \
    --num-prompts 5 \
    --random-input-len 128 \
    --ignore-eos \
    --random-output-len 1024 \    
    --max-concurrency 1 \
    --served_model_name DeepSeek-R1-0528 \
    --save-result \
    --result-dir ./benchmark_result \
    --result-filename result.json     
```
结果说明可参考[性能结果指标说明](#performance)。



本次测试使用“/test/benchmark/benchmark.sh”进行批量测试。


### EvalScope 测试模型性能

通过 EvalScope 框架测试模型性能的指令如下所示，所在路径为容器（测试模型精度的容器）内的“/root/evalscope”目录下。

```shell
python run_performance.py
```
测试模型性能前，需根据实际情况修改以下参数。

- model_name：模型名称。

- url：Vastai vLLM服务地址。

```{code-block}
import csv
import os
from typing import List, Dict, Any
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.arguments import Arguments

from utils import save_to_csv

# 常量定义
MILLISECONDS_PER_SECOND = 1000

# 配置参数 - 建议改为从环境变量或配置文件中读取
model_name = "DeepSeek-R1-0528"
tokenizer_path = f"deepseek-ai/{model_name}"
url = os.getenv("API_URL", "http://127.0.0.1:8000/v1/chat/completions")
api = "openai"
api_key = os.getenv("API_KEY", "token-abc123")  # 从环境变量读取API密钥
dataset = "random"
max_concurrencies = [1, 2, 4]
pre_req_nums = 1
input_tokens_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 54272, 65536]
output_tokens_list = [1024, 8192]
max_seqlen = 65536
max_per_request = 56320
csv_file_path = "evalscope_benchmark.csv"
... ...
... ...
```



本节以 DeepSeek-R1-0528 模型为例进行说明如何通过EvalScope 框架测试模型性能。



**步骤 1.** 启动Vastai vLLM服务。详细说明可参考[启动Vast vLLM 服务](#vastai_vllm)。

**步骤 2.** 新打开一个终端拉取测试模型精度的镜像。

```shell
docker pull harbor.vastaitech.com/ai_deliver/deepseek_eval:latest
```

**步骤 3.** 在新打开的终端运行测试模型精度的容器。

```shell
docker run --ipc=host --rm -it  --network host \
       --name=deepseek_eval harbor.vastaitech.com/ai_deliver/\
       deepseek_eval:latest bash
```


**步骤 4.** 根据实际情况修改精度测试脚本“/root/evalscope/run_performance.py”中的模型名称、Vastai vLLM 服务地址，类似如下所示。


```{code-block}
import csv
import os
from typing import List, Dict, Any
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.arguments import Arguments

from utils import save_to_csv

# 常量定义
MILLISECONDS_PER_SECOND = 1000

# 配置参数
model_name = "DeepSeek-R1-0528"
tokenizer_path = f"deepseek-ai/{model_name}"
url = os.getenv("API_URL", "http://127.0.0.1:8000/v1/chat/completions")
api = "openai"
api_key = os.getenv("API_KEY", "token-abc123")  # 从环境变量读取API密钥
dataset = "random"
max_concurrencies = [1, 2, 4]
pre_req_nums = 1
input_tokens_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 54272, 65536]
output_tokens_list = [1024, 8192]
max_seqlen = 65536
max_per_request = 56320
csv_file_path = "evalscope_benchmark.csv"
... ...
... ...
```




**步骤 5.**  测试 DeepSeek-R1-0528 模型性能。

```shell
cd /root/evalscope
python run_performance.py
```

性能结果保存在`${csv_file_path}`中，结果说明可参考[性能结果指标说明](#performance)。


<a id="performance"></a>
## 性能结果指标说明

- Maximum req： 最大并发数。

- Duration：请求测试耗时。

- Successful req：请求总数。

- input tokens：输入Token数量。

- generated tokens：输出Token数量。

- Req throughput：每秒处理的请求数。

- Output token throughput：每秒输出Token数量。

- Total Token throughput：每秒生成Token数量。

- Mean TTFT ：从用户发送请求到模型生成第一个 Token 的平均时间。

- Mean TPOT：模型生成每个输出 Token 所需的平均时间。

- Decode Token throughput：Decode阶段每秒输出Token数量。

- Per-req Decoding token throughput：Decode阶段平均每个用户每秒输出Token数量。




# 测试模型精度


模型精度测试通过Vastai vLLM 服务加载模型，并使用 EvalScope 框架进行评估。测试指令如下所示，所在路径为容器（测试模型精度的容器）内的“/root/evalscope”目录下。

- 针对 DeepSeek R1系列模型，其执行命令为：
```shell
python run_precision_R1_0528.py
```


测试模型精度前需根据实际情况修改“run_precision_xxx”中如下参数。其中，xxx为R1_0528 或 R1，请根据实际情况替换。

- model：模型名称。

- api_url：Vastai vLLM 服务所在地址。

- max_tokens: 模型最大 Token。

  - DeepSeek-R1:需设置为 32768。

  - DeepSeek-R1-0528：需设置为 61440。



```{code-block} 
# set dataset
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model='DeepSeek-R1-0528',   # 模型名称 (需要与部署时的模型名称一致)
    api_url='http://127.0.0.1:8000/v1',  # 推理服务地址
    api_key='token-abc123',
    eval_type=EvalType.SERVICE,   # 评测类型，SERVICE表示评测推理服务
    datasets=[
        'mmlu_pro',
        'drop', 
        'ifeval', 
        'gpqa', 
        'live_code_bench',
        'aime24', 
        'math_500',
        'ceval',        
        'general_mcq',     # 选择题格式固定为 'general_mcq'
    ],
    ... ...
    ... ...
    generation_config={       # 模型推理配置
        #'max_tokens': 61440,  # 最大生成token数，R1-0528: 61440, R1:32768，V3-0324: 32768, V3: 16384
         'max_tokens':32768   # for R1
        'temperature': 0.6,   # 采样温度 (deepseek 报告推荐值)
        'top_p': 0.95,        # top-p采样 (deepseek 报告推荐值)
        'n': 1                # 每个请求产生的回复数量 (注意 lmdeploy 目前只支持 n=1)
    },
   ... ...
```







本节以 DeepSeek-R1-0528 模型为例进行说明如何测试模型精度。



**步骤 1.** 启动vLLM服务。详细说明可参考[启动Vast vLLM 服务](#vastai_vllm)。

**步骤 2.** 新打开一个终端拉取测试模型精度的镜像。

```shell
docker pull harbor.vastaitech.com/ai_deliver/deepseek_eval:latest
```

**步骤 3.** 在新打开的终端运行测试模型精度的容器。

```shell
docker run --ipc=host --rm -it  --network host \
       --name=deepseek_eval harbor.vastaitech.com/ai_deliver/\
       deepseek_eval:latest bash
```

**步骤 4.** 根据实际情况修改精度测试脚本“/root/evalscope/run_precision_R1_0528.py”中的模型名称、Vastai vLLM服务地址、最大Token，类似如下所示。

```{code-block} 
# set dataset
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model='DeepSeek-R1-0528',   # 模型名称 (需要与部署时的模型名称一致)
    api_url='http://127.0.0.1:8000/v1',  # 推理服务地址
    api_key='token-abc123',
    eval_type=EvalType.SERVICE,    # 评测类型，SERVICE表示评测推理服务
    datasets=[
        'mmlu_pro',
        'drop', 
        'ifeval', 
        'gpqa', 
        'live_code_bench',
        'aime24', 
        'math_500',
        'ceval',   
        'general_mcq',     # 选择题格式固定为 'general_mcq'
    ],
    ... ...
    ... ...
    generation_config={       # 模型推理配置
        #'max_tokens': 32768,  # 最大生成token数，V3-0324: 32768, V3: 16384
        'max_tokens': 16384  # for V3 
        'temperature': 0.6,   # 采样温度 (deepseek 报告推荐值)
        'top_p': 0.95,        # top-p采样 (deepseek 报告推荐值)
        'n': 1                # 每个请求产生的回复数量 (注意 lmdeploy 目前只支持 n=1)
    },
   ... ...
```




**步骤 5.**  测试 DeepSeek-R1-0528 模型精度。

```shell
cd /root/evalscope
python run_precision_R1_0528.py
```

本次测试使用了MMLU、MATH-500、CLUEWSC数据集。其中，MATH-500和CLUEWSC为全量数据集，MMLU仅使用了前10个子集（涵盖哲学、历史、计算机科学等学科）。精度结果如下所示。


其中，“xxx_vacc”表示在瀚博硬件设备上的精度测试结果，“xxx_cuda”表示在NVIDIA上的精度测试结果，xxx为数据集名称。


> 精度测试结果根据ds3_0526版本得到。


![Precis.png](https://storage.vastaitech.com/storage/v1/download/430311395716894720/Precis.png?X_Amz_Algorithm=AES&X_Amz_Credential=None-430311395716894720&X_Amz_Date=2025-07-01T22:31:44Z&X_Amz_Expires=86400&X_Amz_SignedHeaders=host&X_Amz_Signature=40c82f554c668f101924c32c4df1a6340b7cd5a8fa196684f2649a21c8999b3e)




# 启动 Vastai Open WebUI 服务

Vastai Open WebUI通过容器启动，本节以 DeepSeek-R1-0528 模型为例进行说明如何访问 Vastai Open WebUI。




**操作步骤**

**步骤 1.** 启动 Vastai vLLM 服务。详细说明可参考[启动Vast vLLM 服务](#vastai_vllm)。

**步骤 2.** 新打开一个终端拉取 Vastai Open WebUI 镜像。
```shell
docker pull harbor.vastaitech.com/ai_deliver/vast-webui:latest
```

**步骤 3.** 启动 Vastai Open WebUI 服务。

```shell
docker run -d \
    -v vast-webui:/app/backend/data \
    -e ENABLE_OLLAMA_API=False \
    --network=host \
    -e PORT=18080 \
    -e OPENAI_API_BASE_URL="http://127.0.0.1:8000/v1" \
    -e DEFAULT_MODELS="/weights/DeepSeek-R1-0528" \
    -e DEFAULT_LOCALE="cn" \
    --name vast-webui \
    --restart always \
    harbor.vastaitech.com/ai_deliver/vast-webui:latest
```
>上述指令默认在 x86 架构的 CPU 环境中执行。如果 CPU 是 ARM 架构，则`harbor.vastaitech.com/ai_deliver/vast-webui:latest`需替换为`harbor.vastaitech.com/ai_deliver/vast-webui:arm_latest`。


其中，`OPENAI_API_BASE_URL`为 Vastai vLLM 服务地址，`DEFAULT_MODELS`为原始模型权重所在路径。请根据实际情况替换。


Vastai Open WebUI 服务启动后，即可通过[http://HostIP:18080](http://HostIP:18080)访问 Vastai Open WebUI。

其中，“HostIP” 为 Vastai WebUI 服务所在IP地址。




**步骤 4.** 访问 Vastai Open WebUI 页面，并根据界面提示注册账号并登录。

首次进入需设置管理员账号密码。设置完毕后，进入如下所示主页。

![vastai_openwebui.png](https://storage.vastaitech.com/storage/v1/download/430387876463775744/vastai_openwebui.png?X_Amz_Algorithm=AES&X_Amz_Credential=None-430387876463775744&X_Amz_Date=2025-07-01T22:31:44Z&X_Amz_Expires=86400&X_Amz_SignedHeaders=host&X_Amz_Signature=23b5f30b456b551706f5298db999eb2882d3050ea2c1c11bbc93b3f5e3de9808)

> 如果瀚博已提供环境，则默认用户名为“admin@vastai.com”，默认密码为“admin123”。


**步骤 5.** 连接 Vastai vLLM 服务并添加DeepSeek-R1-0528模型。


如果是普通用户，也可在“设置 > 外部连接”页签添加 Vastai vLLM服务和模型，但是添加后仅针对当前普通用户有效。


1. 在“管理员面板 > 设置 > 外部连接”页签的“管理 Open API 连接”栏单击“+”。


![add_vllm.png](https://storage.vastaitech.com/storage/v1/download/430386869646266368/add_vllm.png?X_Amz_Algorithm=AES&X_Amz_Credential=None-430386869646266368&X_Amz_Date=2025-07-01T22:31:44Z&X_Amz_Expires=86400&X_Amz_SignedHeaders=host&X_Amz_Signature=fb30e39081e6db7de4baa38a8449fb7e53af7d7d8a83045b503fa1a61b95db76)

2. 在“添加一个连接”页面配置 Vastai vLLM 服务地址、密钥和DeepSeek-R1-0528模型地址并保存。

- Vastai vLLM 服务地址格式：http://HostIP:Port/v1。其中，HostIP 为 Vastai vLLM 服务所在地址，Port 为 Vastai vLLM 服务端口。

- 密钥：API密钥，需配置为“token-abc123”。

- 模型地址：原始模型权重文件所在路径。



![add_url_model.png](https://storage.vastaitech.com/storage/v1/download/430387014215536640/add_url_model.png?X_Amz_Algorithm=AES&X_Amz_Credential=None-430387014215536640&X_Amz_Date=2025-07-01T22:31:44Z&X_Amz_Expires=86400&X_Amz_SignedHeaders=host&X_Amz_Signature=23b1f290f51727c498613e255e072ec23eb82fe75e15a60e588dd96b5599410a)

3. 在“管理员面板 > 设置 > 界面”页签禁用{numref}`disable_ui_all2`红框中的功能以防止 Vastai Open WebUI 自动调用大模型执行红框中的功能。

:::{figure-md} disable_ui_all2
:class: myclass

![](images/disable_ui.png)


禁止自动调用大模型

:::

![disable_ui.png](https://storage.vastaitech.com/storage/v1/download/430387118813089792/disable_ui.png?X_Amz_Algorithm=AES&X_Amz_Credential=None-430387118813089792&X_Amz_Date=2025-07-01T22:31:44Z&X_Amz_Expires=86400&X_Amz_SignedHeaders=host&X_Amz_Signature=f8ab3dc69b0e85c7f4263ec0b4c9eb933b445aae0ca76759ecbd36b2a7e6d9dd)

**步骤 6.** 开启一个新对话进行简单体验。



![chat.png](https://storage.vastaitech.com/storage/v1/download/430387205324804096/chat.png?X_Amz_Algorithm=AES&X_Amz_Credential=None-430387205324804096&X_Amz_Date=2025-07-01T22:31:44Z&X_Amz_Expires=86400&X_Amz_SignedHeaders=host&X_Amz_Signature=0fa519dd07f6926dc7454c22805b64f1854d96142356cb8c0f163d610ac5601f)

本节仅简单说明如何使用Vastai Open WebUI。详细使用说明可参考[https://openwebui-doc-zh.pages.dev/features/](https://openwebui-doc-zh.pages.dev/features/)。

