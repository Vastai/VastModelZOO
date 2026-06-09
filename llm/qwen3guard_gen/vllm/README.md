# 概述

本文档旨在指导用户如何基于 vLLM 在瀚博硬件设备上部署 Qwen3Guard-Gen 系列模型，以及测试模型的精度和性能。

# 硬件要求

| 模型规格                         | 最低硬件配置要求                   |
| -------------------------------- | ---------------------------------- |
| Qwen3Guard-Gen 系列（BF16） | 单卡 VA16 / 单卡 VA1L / 单卡 VA10L |



## 模型支持

  | model                            | huggingface                                                                                                       | modelscope                                                                                                                  | parameter | dtype | parallel  |
  | :------------------------------- | :---------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------- | :-------- | :---- | :-------- |
  | Qwen3Guard-Gen-0.6B                       | [Qwen/Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B)                                                         | [Qwen/Qwen3Guard-Gen-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3Guard-Gen-0.6B)                                                         | 0.6B      | BF16  | TP1/2/4/8 |
  | Qwen3Guard-Gen-4B                         | [Qwen/Qwen3Guard-Gen-4B](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B)                                                             | [Qwen/Qwen3Guard-Gen-4B](https://www.modelscope.cn/models/Qwen/Qwen3Guard-Gen-4B)                                                             | 4B        | BF16  | TP1/2/4/8 |
  | Qwen3Guard-Gen-8B                         | [Qwen/Qwen3Guard-Gen-8B](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B)                                                             | [Qwen/Qwen3Guard-Gen-8B](https://www.modelscope.cn/models/Qwen/Qwen3Guard-Gen-8B)                                                             | 8B        | BF16  | TP2/4/8   |

## 使用限制

| model                 | parallel  | seq limit                                     | mtp   | tips              |
| :-------------------- | :-------- | :-------------------------------------------- | :---- | :---------------- |
| Qwen3Guard-Gen-0.6B   | TP1/2/4/8 | max-model-len 32768                           | ❌    | max-concurrency 4 |
| Qwen3Guard-Gen-4B     | TP1/2/4/8 | max-model-len 32768                           | ❌    | max-concurrency 4 |
| Qwen3Guard-Gen-8B     | TP2/4/8   | max-model-len 32768                           | ❌    | max-concurrency 4 |



## 模型下载
1. 通过hf-mirror下载

- 参考[hf-mirror](https://hf-mirror.com/)下载权重
  ```shell
  wget https://hf-mirror.com/hfd/hfd.sh
  chmod a+x hfd.sh
  export HF_ENDPOINT=https://hf-mirror.com
  apt install aria2
  ./hfd.sh Qwen/Qwen3Guard-Gen-0.6B -x 10 --local-dir Qwen3Guard-Gen-0.6B
  ```

2. 或通过modelscope下载

- 参考[modelscope](https://modelscope.cn/docs/models/download)下载权重
  ```shell
  pip install modelscope -i https://mirrors.ustc.edu.cn/pypi/web/simple
  export PATH=$PATH:~/.local/bin
  modelscope download --model Qwen/Qwen3Guard-Gen-0.6B --local_dir ./Qwen3Guard-Gen-0.6B
  ```


## 启动模型服务

1. 参考官方启动命令：[vllm](https://docs.vllm.ai/en/latest/cli/#bench)

  ```bash
  docker run \
      -e VACC_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
      -e LLM_MAX_PREFILL_SEQ_LEN="102400" \
      --privileged=true --shm-size=256g \
      --name vllm_service \
      -v /path/to/model:/weights/ \
      -p 8000:8000 \
      --ipc=host \
      harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-26.05 \
      vllm serve /weights/Qwen/Qwen3Guard-Gen-0.6B \
      --trust-remote-code \
      --tensor-parallel-size 2 \
      --max-model-len 32768 \
      --enforce-eager \
      --host 0.0.0.0 \
      --port 8000 \
      --served-model-name Qwen3Guard-Gen-0.6B
  ```

- 参数说明如下
  - `LLM_MAX_PREFILL_SEQ_LEN="102400"`：最大prefill长度环境变量设置。

  - `--tensor-parallel-size`：张量并行数。

  - `--model`：原始模型权重所在路径。请根据实际情况替换。

  - `--port`：模型服务端口。

  - `--served-model-name`：模型名称。

  - `--max-model-len`：模型最大上下文长度。


## 模型性能测试

> 模型性能包含吞吐和推理时延，可通过 vLLM 服务加载模型，并使用 vLLM 自带框架进行性能测试。

1. 参考vLLM文档测试模型性能：[benchmarking/cli](https://docs.vllm.ai/en/latest/benchmarking/cli/)

```shell
vllm bench serve \
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

- 其中，参数说明如下

  - `--host`：vLLM 推理服务所在 IP 地址。

  - `--port`：vLLM 推理服务端口。

  - `--model`：原始模型权重文件所在路径。和 vLLM 推理服务启动时设置的模型路径一致。

  - `--dataset-name`：数据集名称。

  - `--num-prompts`：测试时使用的输入数据数量。

  - `--random-input-len`：输入序列的长度。

  - `--ignore-eos`：用于控制生成文本时是否忽略模型的 EOS（End-of-Sequence） Token，即结束标记，如 `<|endoftext|>` 或 `</s>`。

  - `--random-output-len`： 输出序列的长度。

  - `--max-concurrency`：最大请求并发数。

  - `--served-model-name`：API 中使用的模型名称。
  > 该参数设置应与模型服务启动脚本中`--served-model-name`参数一致。

  - `--save-result`：是否保存测试结果。如果设置该参数，则测试保存至`--result-dir` 和 `--result-filename` 指定的路径。

  - `--result-dir`：测试结果保存目录。如果不设置，则保存至当前路径。

  - `--result-filename`：测试结果文件名称。



2. 测试示例
- 启动vLLM服务
- 测试`Qwen/Qwen3Guard-Gen-0.6B`模型性能

  ```shell
  docker exec -it  vllm_service bash
  cd /test/benchmark
  mkdir benchmark_result
  export OPENAI_API_KEY="token-abc123"
  vllm bench serve \
      --host <IP> \
      --port 8000 \
      --model /weights/Qwen/Qwen3Guard-Gen-0.6B \
      --dataset-name random \
      --num-prompts 3 \
      --random-input-len 128 \
      --ignore-eos \
      --random-output-len 1024 \
      --max-concurrency 1 \
      --served-model-name Qwen3Guard-Gen-0.6B \
      --save-result \
      --result-dir ./benchmark_result \
      --result-filename result.json
  ```

  - 其中，`vllm_service`为 vLLM 服务容器名称，可通过`docker ps |grep vLLM`查询；`host`为本机ip地址。
  - 推荐指定采样参数，控制变量：`--temperature 0.0 --min-p 0.0 --top-k 0 --top-p 1.0`


## 性能结果指标说明

- Maximum request concurrency： 最大并发数。

- Benchmark duration (s)：请求测试耗时。

- Successful requests：请求总数。

- Total input tokens：输入Token数量。

- Total generated tokens：输出Token数量。

- Request throughput：每秒处理的请求数。

- Output token throughput：每秒输出Token数量。

- Total Token throughput：每秒生成Token数量。

- Mean TTFT ：从用户发送请求到模型生成第一个 Token 的平均时间。

- Mean TPOT：模型生成每个输出 Token 所需的平均时间。

- Mean ITL: token间延迟。


## 模型精度测试

### 测试数据集

QwenGuardTest 数据集可从以下平台下载：
- [🤗 HuggingFace](https://huggingface.co/datasets/Qwen/Qwen3GuardTest)
- [🤖 ModelScope](https://modelscope.cn/datasets/Qwen/Qwen3GuardTest)

该数据集包含三个不同的子集：
- thinking：该子集包含 1,059 个样本，涵盖了带有“thinking”过程的回复。这些样本是通过向各类具备“思考”能力的模型输入 Beavertails 测试集中的有害提示词生成的。

- thinking_loc：这是 thinking 子集的一个子集，包含 569 个样本，所有样本均被标记为“Unsafe”。每个样本都标注了首个不安全句子的精确起始和结束索引。

- response_loc：该子集包含 813 个样本，仅包含最终回复，不含思考过程。该子集中的所有样本均被标记为“Unsafe”，并标注了首个不安全句子的起始和结束索引。


### 精度测试步骤

1. 下载 `QwenGuardTest` 数据集；
2. 启动 vLLM 模型服务
3. 参考脚本：
    - [eval_openai.py](./eval/eval_openai.py)，OpenAI 接口的精度测试脚本
    - [eval_transformers.py](./eval/eval_transformers.py) (可选)，[官方](https://github.com/QwenLM/Qwen3Guard/blob/main/eval/eval_gen.py) Transformers 接口的精度测试脚本

  - 测评主要参数 (`eval_openai.py`)：
    | 参数 | 类型 | 必填 | 默认值 | 说明 |
    |------|------|------|--------|------|
    | `--model_name` | `str` | ✅ | - | 服务端部署的模型名称，需与 vLLM `--served-model-name` 一致。 |
    | `--base_url` | `str` | ✅ | - | OpenAI 兼容 API 地址，如 `http://localhost:8000/v1`。 |
    | `--api_key` | `str` | ❌ | `EMPTY` | API 密钥；若服务端未开启鉴权，保持默认即可。 |
    | `--dataset_path` | `str` | ✅ | - | 数据集路径。可为 Hugging Face 名称或本地 `.jsonl` 文件路径。 |
    | `--split_type` | `str` | ❌ | `thinking` | Hugging Face 数据集的分割名称；对本地 JSONL 无效。 |
    | `--output_dir` | `str` | ❌ | `./output` | 结果输出目录，脚本会自动创建。 |
    | `--max_tokens` | `int` | ❌ | `16384` | 模型生成的最大 token 数。 |
    | `--temperature` | `float` | ❌ | `0.0` | 采样温度，建议保持 `0.0` 以保证可复现性。 |
    | `--top_p` | `float` | ❌ | `1.0` | 核采样概率阈值（top_p），与 temperature 共同控制生成多样性。 |

4. 测试示例

```bash
python eval_openai.py \
  --model_name Qwen3Guard-Gen-0.6B \
  --base_url http://localhost:8000/v1 \
  --dataset_path ./data/test.jsonl \
  --output_dir ./output
```

5. 输出说明

- 结果文件
  
  - 脚本会在 `--output_dir` 下生成如下格式的 JSONL 文件：`{output_dir}/eval_{model_name}_{split_type}.jsonl`

- 终端指标

  ```
  ==================================================
  评测结果
  ==================================================
  模型名称 : Qwen3Guard-Gen-0.6B
  数据分割 : thinking
  真实 Unsafe 样本数 : 569
  预测 Unsafe 样本数 : 459
  真正例 (TP)        : 442
  Recall    : 0.7768
  Precision : 0.9630
  F1 Score  : 0.8599
  ==================================================
  ```
