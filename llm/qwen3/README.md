# 概述 

本文档旨在指导用户如何基于 vLLM 和 Open WebUI 在瀚博硬件设备上部署 Qwen3 系列模型，以及测试模型的精度和性能。

# 硬件要求

部署 Qwen3-30B-A3B-FP8 模型进行推理至少需要单卡VA16（1 x 128G）或单卡VA1L(1 x 64G)。

# 版本信息


本次发布软件版本为 [AI3.0_SP7_0728](https://developer.vastaitech.com/downloads/delivery-center?version_uid=440893211821608960)。

>该版本为中期迭代版本，不作为正式出货版本。

## 版本配套说明


| 组件 |  版本|
| --- | --- |
| Driver | V3.3.0|
| torch | 2.7.0+cpu|
| vllm | 0.9.2+cpu|
| vllm_vacc| AI3.0_SP7_0728|

## 支持的模型

当前支持的模型如下所示：

- [Qwen3-30B-A3B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B-FP8)

- [Qwen3-30B-A3B-Instruct-2507-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)

模型下载步骤如下所示。

1. 安装 ModelScope。

```shell
pip install modelscope -i https://mirrors.ustc.edu.cn/pypi/web/simple 
export PATH=$PATH:~/.local/bin
```

2. 根据实际情况选择对应的模型下载。

其中，“`$Path`”为模型保存路径，“`$Model_Name`”为模型名称，如下所示，请根据实际情况替换。

`$Model_Name`：

- Qwen3-30B-A3B-FP8
- Qwen3-30B-A3B-Instruct-2507-FP8


每个模型大小约为31GB,下载时请确保“`$Path`”所在的磁盘存储空间是否足够。


下载过程中如果出现某个文件下载失败的情况，可等命令执行完成后重新执行该命令，继续下载未下载完成的文件。



```shell
modelscope download --model Qwen/$Model_Name --local_dir $Path/$Model_Name
```


## 注意事项

在当前硬件配置下，测试模型性能和精度时需注意以下限制条件：

- Qwen3-30B-A3B-FP8 TP4 最大支持128k上下文，Qwen3-30B-A3B-FP8 TP2 最大支持64k上下文，输入最大支持56k。Qwen3-30B-A3B-Instruct-2507-FP8 最大支持64K上下文，输入最大支持56K。	

- 同时支持最大并发数为 4。

- 对于超过上下文长度的请求，内部会拦截不做处理，需要客户端自行处理。 



# 环境安装


前置依赖说明：[Requirement.md](https://developer.vastaitech.com/downloads/delivery-center?version_uid=440893211821608960)


部署 Qwen3 系列模型支持两种部署方式：

- 一键安装：表示通过脚本一键部署，用户无需再单独安装驱动、启动vLLM 服务。
- 分步安装：需根据操作步骤安装驱动、启动 vLLM 服务。
<a id="install_one_click"></a>
## 一键安装

通过如下命令一键启动 vLLM 服务。命令下载链接：[开发者中心](https://developer.vastaitech.com/downloads/delivery-center?version_uid=440893211821608960)
```shell
./vallmdeploy_AI3.0_SP7_0728.run <Model_Type> <Model_Path>
```

参数说明如下所示。
    
- Model_Type：可设置为 Qwen3-TP2 或 Qwen3-TP4。
    
- Model_Path: 模型权重路径。

注意：一键安装前需要停掉运行中的 vllm_service 和 haproxy-server 

参考命令:
```bash
sudo docker rm -f vllm_service haproxy-server 
```
<a id="install_stepbystep"></a>
## 分步安装

部署 DeepSeek-V3 及 DeepSeek-R1 系列模型前，请确保已从[开发者中心](https://developer.vastaitech.com/downloads/delivery-center?version_uid=440893211821608960)下载配套版本的驱动（Driver）和《PCIe 驱动安装指南》，并按指南完成驱动安装。


<a id="vastai_vllm"></a>
## 启动 vLLM 服务


**步骤 1.** 获取[haproxy](../common/haproxy)包。

假设存放路径为“/home/username”，请根据实际情况替换。


**步骤 2.**  启动 vLLM 服务。

本文档默认使用 deploy.py 启动vllm server。如果需要 vllm server 原生启动方式，可参考 [docker-compose](./docker-compose/)

对于Qwen3-30B-A3B-FP8模型，启动命令：
```shell
cd /home/username/haproxy
python3 deploy.py --instance 8 \
    --tensor-parallel-size 4 \
    --image harbor.vastaitech.com/ai_deliver/vllm_vacc:AI3.0_SP7_0728 \
    --model /home/username/weights/Qwen3-30B-A3B-FP8 \
    --port 8000 \
    --management-port 9000 \
    --max-batch-size-for-instance 4 \
    --served-model-name Qwen3 \
    --max-model-len 65536 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --enable-qwen3-rope-scaling \
    --enable-auto-tool-choice \
    --tool-call-parser hermes 
```

对于Qwen3-30B-A3B-Instruct-2507-FP8模型，启动命令：
```shell
cd /home/username/haproxy
python3 deploy.py --instance 8 \
    --tensor-parallel-size 4 \
    --image harbor.vastaitech.com/ai_deliver/vllm_vacc:AI3.0_SP7_0728 \
    --model /home/username/weights/Qwen3-30B-A3B-FP8 \
    --port 8000 \
    --management-port 9000 \
    --max-batch-size-for-instance 4 \
    --served-model-name Qwen3 \
    --max-model-len 65536 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes 
```

参数说明如下所示。
    
- `--instance`： 模型推理实例。

- `--tensor-parallel-size`：张量并行数, 目前Qwen3 系列支持 TP2 对应参数：“--tensor-parallel-size 2” 此时 “--instance” 最大支持16； TP4 对应参数：“--tensor-parallel-size 4”，此时 “--instance” 最大支持8。

- `--image`：模型服务镜像。

- `--model`：原始模型权重所在路径。请根据实际情况替换。

- `--port`：模型服务端口。

- `--management-port`：管理端口。

- `--max-batch-size-for-instance`：每个实例的最大 Batch Size。

- `--served-model-name`：模型名称。仅支持设置为 Qwen3。

- `--max-model-len`：模型最大上下文长度，Qwen3-30B-A3B-FP8 TP4 最大支持128k上下文，Qwen3-30B-A3B-FP8 TP2 最大支持64k上下文，Qwen3-30B-A3B-Instruct-2507-FP8 最大支持64K上下文。

- `--enable-reasoning`：是否启动模型推理内容生成功能。需与`--reasoning-parser`参数配套使用。

- `--reasoning-parser`：指定用于从模型输出中提取推理内容的推理解析器。

- `--enable-qwen3-rope-scaling`：是否启动 Qwen3 模型的 RoPE 缩放功能，使模型最大上下文长度支持 64K。

- `--enable-auto-tool-choice`：启用自动工具选择功能，使模型能够根据用户输入自动决定是否需要调用工具（如 API、函数），并选择最合适的工具。

- `--tool-call-parser`：设置工具调用解析器，用于解析模型的输出中是否包含工具调用请求，并将其转换为结构化格式（如 JSON）。对于Qwen3系列模型，需设置为 hermers。

- `--chat-template`: 指定聊天对话的模板格式, 对于 Qwen3 系列模型，可通过指定参数 “--chat-template  /workspace/qwen3_nonthinking.jinja” 关闭思考模式，同时此时需去掉“--enable-reasoning --reasoning-parser deepseek_r1” 参数才能使关闭思考模式生效。

启动完成后显示如下类似信息。
```shell
Deployment configuration updated successfully.
Docker containers started successfully.
All instances are up and running
```



**步骤 3.** 查看 vLLM 服务的输出日志。

```bash
tail -f llm_serve_0.log
```


**步骤 4.** （可选）停止 vLLM 服务。

如果需停止服务，可执行该步骤。
[docker-compose.yaml](../common/haproxy/docker-compose.yaml)
```bash
docker-compose -f docker-compose.yaml down
```


# 测试模型性能

模型性能包含吞吐和推理时延，可通过 vLLM 服务加载模型，并使用 vLLM 自带框架进行性能测试。

## vLLM 自带框架测试模型性能

通过 vLLM 自带框架进行模型测试的指令如下所示，所在路径为容器（启动 vLLM 服务的容器）内的“/test/benchmark”目录下。

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
    --served-model-name <model_name> \
    --save-result \
    --result-dir <result> \
    --result-filename <result_name>
```


其中，参数说明如下所示。


- `--host`：vLLM 推理服务所在 IP 地址。

- `--port`：vLLM 推理服务端口，需在“qwen3-30b-docker-compose.yaml”中查看确认。

- `--model`：原始模型权重文件所在路径。和 vLLM 推理服务启动时设置的模型路径一致。

- `--dataset-name`：数据集名称。

- `--num-prompts`：测试时使用的输入数据数量。

- `--random-input-len`：输入序列的长度。

- `--ignore-eos`：用于控制生成文本时是否忽略模型的 EOS（End-of-Sequence） Token，即结束标记，如 `<|endoftext|>` 或 `</s>`。

- `--random-output-len`： 输出序列的长度。

- `--max-concurrency`：最大请求并发数。

- `--served-model-name`：API 中使用的模型名称。
  - 如果通过一键安装启动vLLM 服务， 该参数设置应与<Model_Type>一致，设置为 Qwen3-TP2 或 Qwen3-TP4 
  
  - 如果是通过分步安装启动vLLM 服务，该参数设置应与deploy.py 启动脚本中“--served-model-name” 参数一致

- `--save-result`：是否保存测试结果。如果设置该参数，则测试保存至`--result-dir` 和 `--result-filename` 指定的路径。

- `--result-dir`：测试结果保存目录。如果不设置，则保存至当前路径。

- `--result-filename`：测试结果文件名称。


本节以 Qwen3-30B-A3B-FP8 模型为例进行说明如何测试模型性能。

**步骤 1.** 启动 vLLM 服务。

**步骤 2.** 测试Qwen3-30B-A3B-FP8模型性能。

```shell
docker exec -it  vllm_service bash
cd /test/benchmark
mkdir benchmark_result
export OPENAI_API_KEY="token-abc123"
python3 benchmark_serving.py \
    --host <IP> \
    --port 8000 \
    --model /weights/Qwen3-30B-A3B-FP8 \
    --dataset-name random \
    --num-prompts 5 \
    --random-input-len 128 \
    --ignore-eos \
    --random-output-len 1024 \    
    --max-concurrency 1 \
    --served-model-name Qwen3 \
    --save-result \
    --result-dir ./benchmark_result \
    --result-filename result.json     
```
其中，“vllm_service”为 vLLM 服务容器名称，可通过`docker ps |grep vLLM`查询；“host”为本机ip地址。


本次测试使用“/test/benchmark/benchmark.sh”进行批量测试。


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


模型精度测试通过 vLLM 服务加载模型，并使用 vaeval 进行评估。vaeval 工具基于 EvalScope 二次开发，EvalScope 说明可参考[EvalScope 用户手册](https://evalscope.readthedocs.io/zh-cn/latest/index.html)。

EvalScope 支持基于原生数据集进行精度测试，也支持基于自定义数据集进行测试。不同的数据集其精度测试配置文件不同。

使用原生数据集进行精度测试，配置文件如下所示，单击[config_eval_qwen3.yaml](./config/config_eval_qwen3.yaml)获取。EvalScope支持的原生数据集可参考[EvalScope支持的数据集](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html)。

```yaml
# vaeval 评估配置文件
model: "Qwen3"
api_url: "http://localhost:8000/v1/chat/completions"
api_key: "EMPTY"
eval_type: "service"
work_dir: "./outputs_eval_qwen3"

datasets:
  - "mmlu_pro"
  - "drop"
  - "ifeval"
  - "gpqa"
  - "live_code_bench"
  - "aime24"
  - "math_500"
  - "ceval"

dataset_args:
  mmlu_pro:
    subset_list: ["computer science", "math", "chemistry", "engineering", "law"]
  gpqa:
    subset_list: ["gpqa_diamond"]
  ceval:
    subset_list: ["computer_network", "operating_system", "computer_architecture", "college_programming", "college_physics"]

eval_batch_size: 32

generation_config:
  max_tokens: 61440
  temperature: 0.6
  top_p: 0.95
  n: 1

stream: true
timeout: 6000000
limit: 50                             
```

参数说明如下所示。
- model：模型名称。
  - 如果通过一键安装启动vLLM 服务， 该参数设置应与<Model_Type>一致，设置为 Qwen3-TP2 或 Qwen3-TP4 
  
  - 如果是通过分步安装启动vLLM 服务，该参数设置应与deploy.py 启动脚本中“--served-model-name” 参数一致


- api_url：vLLM 服务地址。

- api_key：API 密钥。默认值：Empty。

- eval_type：评测类型，设置为service。


- work_dir：评测结果保存路径。

- datasets：数据集名称。支持输入多个数据集，数据集将自动从modelscope下载。

- dataset_args：数据集参数

  - subset_list：评测数据子集列表，指定后将只使用子集数据。

- eval_batch_size：评测批次大小。

- generation_config：生成参数。

  - max_tokens：生成的最大Token数量。

  - temperature：生成温度。

  - top_p：生成top-p。

   - n： 生成序列数量。

- stream：是否使用流式输出，默认值：false。

- timeout：请求超时时间。

- limit：每个数据集最大评测数据量，不填写则默认为全部评测，可用于快速验证。





使用自定义数据集进行精度测试，配置文件如下所示，单击[config_eval_general_mcq_qwen3](./config/config_eval_general_mcq_qwen3.yaml)获取。自定义数据集格式要求可参考[大语言模型自定义评测数据集](https://evalscope.readthedocs.io/zh-大语言模型自定义评测数据集cn/latest/advanced_guides/custom_dataset/llm.html)。

```yaml
model: Qwen3
api_url: http://localhost:8000/v1/chat/completions
api_key: EMPTY
eval_type: service
datasets:
  - general_mcq
dataset_args:
  general_mcq:
    local_path: "/path/to/cluewsc_custom"
    subset_list:
      - "cluewsc"
    prompt_template: "以下问题的答案有AB两个选项，选出正确答案，请直接回答A或B\n\n{query}"
    eval_split: 'test'
generation_config:
  max_tokens: 61440
  temperature: 0.6
  top_p: 0.95
  n: 1
eval_batch_size: 32
limit: 50
stream: true
timeout: 6000000  
work_dir: ./outputs_eval_qwen3                                            
```

参数说明如下所示。

- model：模型名称。
  - 如果通过一键安装启动vLLM 服务， 该参数设置应与<Model_Type>一致，设置为 Qwen3-TP2 或 Qwen3-TP4 
  
  - 如果是通过分步安装启动vLLM 服务，该参数设置应与deploy.py 启动脚本中“--served-model-name” 参数一致

- api_url：vLLM 服务地址。

- api_key：API密钥。

- eval_type：评测类型，设置为service。


- datasets：自定义数据集名称

- dataset_args：数据集参数


   - general_xxx：自定义数据集名称，根据实际情况替换。

   - local_path：自定义数据集路径。

   - subset_list：自定义数据集子集名称。

   - prompt_template：Prompt模板，

   - eval_split：评测数据集划分。

- generation_config：生成参数

  - max_tokens：生成的最大Token数量。

  - temperature：生成温度。

  - top_p：生成top-p。

   - n： 生成序列数量。

- eval_batch_size：评测批次大小。

- limit：每个数据集最大评测数据量，不填写则默认为全部评测，可用于快速验证。

- stream：是否使用流式输出，默认值：false。

- timeout：请求超时时间。

- work_dir：评测结果保存路径。

本节以 Qwen3-30B-A3B-FP8 模型为例进行说明如何测试模型精度，其中数据集使用原生数据集。

**步骤 1.** 单击[config_eval_qwen3.yaml](./config/config_eval_qwen3.yaml)下载精度配置文件。

假设下载后目录为“/home/username”目录，请根据实际情况替换。

**步骤 2.** 启动 vLLM 服务。

**步骤 3.** 新打开一个终端拉取测试模型精度的镜像。

```shell
docker pull harbor.vastaitech.com/ai_deliver/vaeval:0.1 
```
>上述指令默认在 x86 架构的 CPU 环境中执行。如果 CPU 是 ARM 架构，则`harbor.vastaitech.com/ai_deliver/vaeval:0.1`需替换为`harbor.vastaitech.com/ai_deliver/vaeval:latest_arm`。
>
**步骤 4.** 在新打开的终端运行测试模型精度的容器。

```shell
docker run --ipc=host -it --ipc=host --privileged \
      --name=vaeval -v /home/username:/data harbor.vastaitech.com/ai_deliver/vaeval:0.1 bash
```
其中，“/home/username”为精度测试配置文件所在目录，请根据实际情况替换。

**步骤 5.** 根据实际情况修改精度测试配置文件。

```yaml
# vaeval 评估配置文件
model: "Qwen3"
api_url: "http://localhost:8000/v1/chat/completions"
api_key: "EMPTY"
eval_type: "service"
work_dir: "./outputs_eval_qwen3"

datasets:
  - "mmlu_pro"
  - "drop"
  - "ifeval"
  - "gpqa"
  - "live_code_bench"
  - "aime24"
  - "math_500"
  - "ceval"

dataset_args:
  mmlu_pro:
    subset_list: ["computer science", "math", "chemistry", "engineering", "law"]
  gpqa:
    subset_list: ["gpqa_diamond"]
  ceval:
    subset_list: ["computer_network", "operating_system", "computer_architecture", "college_programming", "college_physics"]

eval_batch_size: 32

generation_config:
  max_tokens: 61440
  temperature: 0.6
  top_p: 0.95
  n: 1

stream: true
timeout: 6000000
limit: 50                    
```




**步骤 6.**  测试 Qwen3-30B-A3B-FP8 模型精度。

```shell
conda activate vaeval
cd /data
vaeval eval config_eval_qwen3.yaml
```

本次测试使用了CLUEWSC、AIME24、CEVAL、DROP等数据集。精度结果如下所示。
其中，“Score_VACC”表示在瀚博硬件设备上的精度测试结果，“Score_NV”表示在NVIDIA上的精度测试结果。


| Model | Dataset |Subset_Num|Sample_Num|Score_VACC|Score_NV|
| --- | --- |--- |--- |--- |--- |
| Qwen3-30B-A3B-FP8 |cluewsc  |1|50|0.92|0.9044|
| Qwen3-30B-A3B-FP8 |aime24  |1|30|0.8|0.8|
| Qwen3-30B-A3B-FP8 |ceval  |5|115|0.913|0.9217|
| Qwen3-30B-A3B-FP8 |drop  |1|50|0.86|0.86|
| Qwen3-30B-A3B-FP8 |gpqa  |1|50|0.62|0.62|
| Qwen3-30B-A3B-FP8 |ifeval  |1|50|0.92|0.92|
| Qwen3-30B-A3B-FP8 |live_code_bench  |1|50|0.98|0.96|
| Qwen3-30B-A3B-FP8 |math_500  |5|243|0.9465|0.9547|
| Qwen3-30B-A3B-FP8 |mmlu_pro  |5|250|0.792|0.796|


# 启动 Open WebUI 服务

Open WebUI通过容器启动，本节以 Qwen3-30B-A3B-FP8 模型为例进行说明如何访问 Open WebUI。




**操作步骤**

**步骤 1.** 启动 vLLM 服务。

**步骤 2.** 新打开一个终端拉取 Open WebUI 镜像。
```shell
docker pull harbor.vastaitech.com/ai_deliver/vast-webui:latest
```

**步骤 3.** 启动 Open WebUI 服务。

```shell
docker run -d \
    -v vast-webui:/app/backend/data \
    -e ENABLE_OLLAMA_API=False \
    --network=host \
    -e PORT=18080 \
    -e OPENAI_API_BASE_URL="http://127.0.0.1:8000/v1" \
    -e DEFAULT_MODELS="/weights/Qwen3-30B-A3B-FP8" \
    -e DEFAULT_LOCALE="cn" \
    --name vast-webui \
    --restart always \
    harbor.vastaitech.com/ai_deliver/vast-webui:latest
```
>上述指令默认在 x86 架构的 CPU 环境中执行。如果 CPU 是 ARM 架构，则`harbor.vastaitech.com/ai_deliver/vast-webui:latest`需替换为`harbor.vastaitech.com/ai_deliver/vast-webui:latest_arm`。


其中，`OPENAI_API_BASE_URL`为 vLLM 服务地址，`DEFAULT_MODELS`为原始模型权重所在路径。请根据实际情况替换。


Open WebUI 服务启动后，即可通过[http://HostIP:18080](http://HostIP:18080)访问  Open WebUI。

其中，“HostIP” 为 Open WebUI 服务所在IP地址。




**步骤 4.** 访问 Open WebUI 页面，并根据界面提示注册账号并登录。

首次进入需设置管理员账号密码。设置完毕后，进入如下所示主页。


![vastai_openwebui.png](../../images/llm/deepseek_r1/vastai_openwebui.png)

> 如果瀚博已提供环境，则默认用户名为“admin@vastai.com”，默认密码为“admin123”。


**步骤 5.** 连接 vLLM 服务并添加Qwen3-30B-A3B-FP8模型。


如果是普通用户，也可在“设置 > 外部连接”页签添加 vLLM服务和模型，但是添加后仅针对当前普通用户有效。


1. 在“管理员面板 > 设置 > 外部连接”页签的“管理 Open API 连接”栏单击“+”。


![add_vllm.png](../../images/llm/deepseek_r1/add_vllm.png)

2. 在“添加一个连接”页面配置 vLLM 服务地址、密钥和Qwen3-30B-A3B-FP8模型地址并保存。

-  vLLM 服务地址格式：http://HostIP:Port/v1。其中，HostIP 为 vLLM 服务所在地址，Port 为 vLLM 服务端口。

- 密钥：API密钥，需配置为“token-abc123”。

- 模型地址：原始模型权重文件所在路径。



![add_url_model.png](../../images/llm/deepseek_r1/add_url_model.png)

3. 在“管理员面板 > 设置 > 界面”页签禁用下图红框中的功能以防止 Open WebUI 自动调用大模型执行红框中的功能。



![disable_ui.png](../../images/llm/deepseek_r1/disable_ui.png)

**步骤 6.** 开启一个新对话进行简单体验。



![chat.png](../../images/llm/deepseek_r1/chat.png)

本节仅简单说明如何使用 Open WebUI。详细使用说明可参考[https://openwebui-doc-zh.pages.dev/features/](https://openwebui-doc-zh.pages.dev/features/)。

