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

### step.3 模型推理

1. 参考瀚博训推软件生态链文档，获取`vllm_vacc`和`torch_vacc`
2. vllm docker
    ```bash
    # 基于开源vllm，构建cpu版本docker镜像
    git clone https://github.com/vllm-project/vllm.git
    cd vllm && git checkout v0.9.2 && docker build -f Dockerfile.cpu -t vllm-cpu-env:v0.9.2 --shm-size=32g . --no-cache && cd ..
    ```
3. vacc docker
    ```bash
    # 基于vllm-cpu-env:v0.9.2基础镜像，构建vllm_vacc镜像
    cd ./torch_vacc/docker && docker build -t vllm_vacc -f Dockerfile --shm-size=16g . --no-cache
    # 或在harbor拉取最新vllm_vacc镜像：https://harbor.vastaitech.com/harbor/projects/158/repositories/vllm_vacc/artifacts-tab

    # 启动容器
    docker run --ipc=host --rm -it --shm-size=256g  --network host --privileged \
        -v /nfs/workspace:/test \
        --name=vllm_vacc \
        vllm_vacc bash

    vllm serve /test/weights/Tongyi-DeepResearch-30B-A3B-FP8 \
    --trust-remote-code --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager \
    --reasoning-parser qwen3 \
    --host 10.24.73.25 --port 8000

    # 超过32k，使用YARN
    --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' --max-model-len 65536

    # 函数调用，tokenizer_config.json中的聊天模板已经包含了对Hermes风格工具调用的支持
    --enable-auto-tool-choice --tool-call-parser hermes
    ```

### step.3 性能测试

1. Online Benchmark

    ```bash
    python3 /workspace/vllm/benchmarks/benchmark_serving.py \
        --model /test/weights/Tongyi-DeepResearch-30B-A3B-FP8 \
        --endpoint /v1/chat/completions \
        --dataset-name sonnet \
        --sonnet-input-len 1024 \
        --sonnet-prefix-len 32 \
        --sonnet-output-len 1024 \
        --dataset-path /workspace/vllm/benchmarks/sonnet.txt \
        --num-prompts 100 \
        --max-concurrency 4 \
        --result-dir /workspace/models/benchmark_result \
        --result-filename result.json \
        --save-result
    ```

2. Offline Throughput Benchmark

    ```bash
    python3 benchmark_throughput.py --model /test/weights/Tongyi-DeepResearch-30B-A3B-FP8 \
        --trust-remote-code --tensor-parallel-size 2 --enforce-eager --max-model-len 32768 \
        --dataset-name random --num-prompts 10  --input-len 1000  --output-len 1000
    ```

3. Offline Latency Benchmark

    ```bash
    python3 benchmark_latency.py --model /test/weights/Tongyi-DeepResearch-30B-A3B-FP8 \
        --trust-remote-code --tensor-parallel-size 2 --enforce-eager --max-model-len 32768 \
        --input-len 1024 --output-len 1024 --batch-size 1 --num-iters 10
    ```

