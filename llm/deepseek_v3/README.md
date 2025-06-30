# DeepSeek-V3

- Technical Report
    - [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

- Huggingface
    - https://hf-mirror.com/deepseek-ai


## Model Arch

![deepseek_v3_arch](../../images/llm/deepseek_v3_arch.png)

### DeepSeek-V3
- 多头潜在注意力机制（Multi-Head Latent Attention, MLA）是为解决MHA在高计算成本和KV缓存方面的局限性而提出的改进。其核心思想是通过低秩联合压缩技术优化键值矩阵，减少内存消耗并提高推理效率。具体特点如下：
    - 低秩联合压缩：MLA通过低秩矩阵分解技术，将键（Key）和值（Value）矩阵压缩为低维的潜在向量（latent vector），从而显著减少存储需求。
    - 优化KV缓存：在推理阶段，MLA仅需存储压缩后的潜在向量，而不是独立的Key和Value矩阵。这使得KV缓存减少了93.3%，显著降低了内存占用。
    - 计算复杂度：通过低秩压缩，MLA在潜在空间中执行注意力计算，降低了计算复杂度，同时保持了较高的特征表达能力。
    - 恢复能力：尽管KV缓存被压缩，MLA仍能通过映射恢复完整的键和值，保持了较高的特征表达能力。

- 混合专家模块（Mixture of Experts，MOE）的核心思想是将任务分解，由多个“专家”子网络（Expert）协同处理，而非全部参数参与每次计算。
    - 专家（Experts）：多个独立的子网络（通常是前馈神经网络），每个专家擅长处理特定类型的数据。
    - 门控机制（Gating Network）：根据输入动态分配权重，决定哪些专家被激活。
    - 稀疏性（Sparsity）：每次仅少数专家参与计算，其余专家处于“休眠”状态。
    - 高效路由策略
        1. Top-K 路由：每个输入 token 仅选择权重最高的前 K 个专家（如 K=2），其余专家不参与计算。
        2. 负载均衡：通过辅助损失（Auxiliary Loss）避免某些专家被过度激活或闲置，确保资源均衡利用。
    - 专家并行（Expert Parallelism）
        1. 分布式计算：专家网络可分布在不同的GPU或计算节点上，通过并行化处理提升吞吐量。
        2. 通信优化：减少专家间的数据传输开销（如通过All-to-All通信高效聚合结果）。

- 多token预测（Multi-Token Prediction，MTP）：通过解码阶段的优化，将1-token的生成，转变成multi-token的生成，从而提升训练和推理的性能。具体来说，在训练阶段，一次生成多个后续token，可以一次学习多个位置的label，进而有效提升样本的利用效率，提升训练速度；在推理阶段并行预估多个token，实现成倍的推理加速来提升推理性能。
- 训练架构
    - HAI-LLM：高效、轻量级的训练框架，其设计充分考虑了多种并行策略，包括DP、PP、TP、EP和FSDP的并行模式
    - FP8：使用FP8来提高计算速度并减少训练期间的显存使用量，在细粒度量化、在线量化、提高累加精度和低精度/混合精度存储与通信等方向作了优化
    - DualPipe：实现高效的流水线并行性，与现有的流水线并行（PP）方法相比，DualPipe具备以下优势：
        - DualPipe的流水线气泡更少，信道使用效率更高
        - DualPipe将前向和后向传播中的计算和通信重叠，解决了跨节点专家并行（EP）带来的繁重通信开销问题
        - 在确保计算与通信比例恒定的情况下，具有很好的Scale-out能力。
    - All-to-All通信与显存优化：制了高效的跨节点All-to-All通信内核，以充分利用IB和NVLink带宽，并节约流式多处理器（Stream Multiprocessor，SM）。DeepSeek还优化了显存分配，以在不使用或少使用张量并行（TP）的情况下训练 V3/R1

### DeepSeek-V3-0324

- 升级版MoE架构：参数扩展至6850亿，采用FP8精度训练，计算效率提升100%
- 技术融合：整合R1的GRPO算法与1.2亿推理链数据，基于V3微调，增量成本$0.4M
- 动态偏差路由：结合节点限制技术，通信流量压缩至传统MoE的1/3，推理速度较V3提升1.8倍
- 专用能力优化：增强数学推理与代码生成能力；中文能力更强


## Pytorch Deploy

### step.1 模型准备
| models  |tips |
| :--- | :--: |
| [deepseek-ai/DeepSeek-V3-Base](https://hf-mirror.com/deepseek-ai/DeepSeek-V3-Base)  | MOE，MLA |
| [deepseek-ai/DeepSeek-V3](https://hf-mirror.com/deepseek-ai/DeepSeek-V3)  | MOE，MLA |
| [deepseek-ai/DeepSeek-V3-0324](https://hf-mirror.com/deepseek-ai/DeepSeek-V3-0324)  | MOE，MLA |


### step.2 模型推理

1. 参考瀚博训推软件生态链文档，`vllm_vacc`和`torch_vacc`: [vastai_software.md](../../../docs/vastai_software.md)
2. vllm docker
    ```bash
    # 基于开源vllm，构建cpu版本docker镜像
    git clone https://github.com/vllm-project/vllm.git
    cd vllm && git checkout v0.7.2 && docker build -f Dockerfile.cpu -t vllm-cpu-env:v0.7.2 --shm-size=32g . --no-cache && cd ..
    ```
3. vacc docker

    ```bash
    # 基于vllm-cpu-env:v0.7.2基础镜像，构建vllm_vacc镜像
    cd ./docker && docker build -t vllm_vacc -f Dockerfile --shm-size=16g . --no-cache

    # 启动容器
    docker run --ipc=host --rm -it --shm-size=256g  --network host --privileged \
        -v /nfs/workspace:/test \
        --name=vllm_vacc \
        vllm_vacc bash

    # 启动vllm服务，使用tp32并行推理
    vllm serve /test/weights/DeepSeek-V3 \
    --trust-remote-code --tensor-parallel-size 32 --max-model-len 16384 --enforce-eager \
    --host 10.24.73.25 --port 8000
    ```

### step.3 性能测试

1. Online Benchmark

    ```bash
    python3 /workspace/vllm/benchmarks/benchmark_serving.py \
        --model /test/weights/DeepSeek-V3 \
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
    python3 benchmark_throughput.py --model /test/weights/DeepSeek-V3 \
        --trust-remote-code --tensor-parallel-size 32 --enforce-eager --max-model-len 16384 \
        --dataset-name random --num-prompts 10  --input-len 1000  --output-len 1000
    ```

3. Offline Latency Benchmark

    ```bash
    python3 benchmark_latency.py --model /test/weights/DeepSeek-V3 \
        --trust-remote-code --tensor-parallel-size 32 --enforce-eager --max-model-len 16384 \
        --input-len 1024 --output-len 1024 --batch-size 1 --num-iters 10
    ```


### step.4 精度测试
1. 基于[evalscope](https://evalscope.readthedocs.io/zh-cn/latest/get_started/introduction.html)工具，测评模型精度
    ```bash
    conda create -n evalscope python=3.10
    conda activate evalscope

    pip install 'evalscope[all]'
    ```

2. 执行测评
    - 使用前述已启动的vllm openapi服务
    - 配置数据集和模型服务: 
        - [eval_ds.py](../common/eval/eval_ds.py)：可选'mmlu_pro','drop', 'ifeval', 'gpqa', 'live_code_bench','aime24', 'math_500','ceval'等数据集，其它支持数据集参见：[LLM评测集](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html#id1)
        - [eval_ds_cluewsc.py](../common/eval/eval_ds_cluewsc.py)：cluewsc数据集原生不支持，通过[custom_dataset](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html)特性转换数据，基于general_mcq测评

    - 执行脚本
        ```bash
        python ../common/eval/eval_ds.py
        python ../common/eval/eval_ds_cluewsc.py
        ```
