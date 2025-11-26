# Tongyi_DeepResearch

- [Tech Blog](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)
- https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B
- https://www.modelscope.cn/models/iic/Tongyi-DeepResearch-30B-A3B

## Vllm Deploy
### step.1 模型准备

| hf  | arch tips | deploy tips |
| :--- | :--: | :--: |
[Alibaba-NLP/Tongyi-DeepResearch-30B-A3B-FP8](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)   | MOE，GQA，thinking |  VA1L/VA10L/VA16，TP2/4 |

> `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`原始权重为BF16格式，暂未提供FP8格式；可按照下述流程量化至FP8格式，以支持VLLM_VACC

### step.2 Quant FP8

- 参考官方安装：[llm-compressor](https://github.com/vllm-project/llm-compressor)
- 执行脚本：[quant.py](./source_code/quant_fp8/quant.py)
    - 相比原生llm-compressor，此脚本做了如下修改，以支持vllm
        - 调整生成量化模型文件夹中config.json字段，形同[config.json#L38](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8/blob/main/config.json#L38)
        - 调整模型state_dict中的量化层键名："weight_scale" -> "weight_scale_inv"

- 执行命令：

    ```bash
    python3 quant.py --model_path Tongyi-DeepResearch-30B-A3B \
        --save_path quantized_model \
        --do_sample
    ```

### step.3 模型推理

```bash
docker run \
    -e VACC_VISIBLE_DEVICES=0,1 \
    --privileged=true --shm-size=256g \
    -v /path/to/model:/weights/ \
    -p 8000:8000 \
    --ipc=host \
    harbor.vastaitech.com/ai_deliver/vllm_vacc:AI3.1.1_GR_1031 \
    vllm serve /weights/Tongyi-DeepResearch-30B-A3B-FP8 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --enforce-eager \
    --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Tongyi-DeepResearch
```

参数说明如下所示。

- `--tensor-parallel-size`：张量并行数，支持设置为2或4。

- `--model`：原始模型权重所在路径。请根据实际情况替换。

- `--port`：模型服务端口。

- `--served-model-name`：模型名称。

- `--max-model-len`：模型最大上下文长度，TP4 最大支持128k上下文，TP2 最大支持64k上下文

- `--rope-scaling`：是否启动 Qwen3 模型的 RoPE 缩放功能，使模型最大上下文长度超过32K。


### step.4 性能测试

模型性能包含吞吐和推理时延，可通过 vLLM 服务加载模型，并使用 vLLM 自带框架进行性能测试。

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
    --server-num <server_num> \
    --save-result \
    --result-dir <result> \
    --result-filename <result_name>
```


其中，参数说明如下所示。


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
> 该参数设置应与模型服务启动脚本中“--served-model-name” 参数一致。

- `--save-result`：是否保存测试结果。如果设置该参数，则测试保存至`--result-dir` 和 `--result-filename` 指定的路径。

- `--result-dir`：测试结果保存目录。如果不设置，则保存至当前路径。

- `--result-filename`：测试结果文件名称。

- `--server-num`: 服务数单服务填 1； 多服务则与 `--instance` 参数设置一致


本节以 Tongyi-DeepResearch-30B-A3B-FP8 模型为例进行说明如何测试模型性能。

**步骤 1.** 启动 vLLM 服务。

**步骤 2.** 测试Tongyi-DeepResearch-30B-A3B-FP8模型性能。

```shell
docker exec -it  vllm_service bash
cd /test/benchmark
mkdir benchmark_result
export OPENAI_API_KEY="token-abc123"
python3 benchmark_serving.py \
    --host <IP> \
    --port 8000 \
    --model /weights/Tongyi-DeepResearch-30B-A3B-FP8 \
    --dataset-name random \
    --num-prompts 5 \
    --random-input-len 128 \
    --ignore-eos \
    --random-output-len 1024 \
    --max-concurrency 1 \
    --served-model-name Tongyi-DeepResearch \
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

