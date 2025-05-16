# vLLM for VACC

- 参考瀚博训推软件生态链文档，`vllm_vacc`和`torch_vacc`: [vastai_software.md](../../../docs/vastai_software.md)

## 软件安装

1. vllm docker
    ```bash
    # 基于开源vllm，构建cpu版本docker镜像
    git clone https://github.com/vllm-project/vllm.git
    cd vllm && git checkout v0.7.2 && docker build -f Dockerfile.cpu -t vllm-cpu-env:v0.7.2 --shm-size=32g . --no-cache && cd ..
    ```

2. vacc docker

    ```bash
    # 基于vllm-cpu-env:v0.7.2基础镜像，构建vllm_vacc镜像
    cd ./docker && docker build -t vllm_vacc -f Dockerfile --shm-size=16g . --no-cache

    # 启动容器
    docker run --ipc=host --rm -it --shm-size=256g  --network host --privileged \
            -v /nfs/workspace:/test \
            --name=vllm_vacc \
            vllm_vacc bash

    # 启动vllm服务，使用tp32并行推理
    taskset -c 20-83 vllm serve /test/weights/DeepSeek-V3 --trust-remote-code --tensor-parallel-size 32 --max-model-len 16384 --enforce-eager --host 10.24.73.25 --port 8000
    
    # 注意R1模型设置--enable-reasoning --reasoning-parser deepseek_r1
    # taskset -c 20-83 vllm serve /test/weights/DeepSeek-R1 --trust-remote-code --tensor-parallel-size 32 --max-model-len 32768 --enforce-eager --enable-reasoning --reasoning-parser deepseek_r1 --host 10.24.73.25 --port 8000
    ```

## 性能测试

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


## 精度测试
1. OpenCompass
    ```bash
    conda create -n opencompass python=3.10
    conda activate opencompass

    git clone https://github.com/open-compass/opencompass.git # git clone https://ghfast.top/https://github.com/open-compass/opencompass.git
    cd opencompass && git checkout 0.4.1 && pip install -e ".[full]" -i https://mirrors.aliyun.com/pypi/simple/
    ```

2. Datasets
    ```bash
    # Download dataset to opencompass/data folder
    # wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
    # wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip

    unzip OpenCompassData-core-20240207.zip
    unzip -o math.zip  -d ./data/

    # some datasets can automatic download
    # pip install ModelScope
    # export DATASET_SOURCE=ModelScope # if not use auto download: unset DATASET_SOURCE
    ```

3. Eval

    - 使用前述已启动的vllm openapi服务
    - 配置数据集和模型服务: [eval/eval_ds.py](./eval/eval_ds.py)

    ```bash
    # 设置精度测试数据集: cluewsc + math_500 + mmlu
    opencompass eval/eval_ds.py --dry-run  # 校验数据集是否完整
    opencompass eval/eval_ds.py            # 启动测试
    ```

    ```
    # 精度测试结果，输出格式类似如下

    | dataset | version | metric | mode | DeepSeek-V3-va16 |
    |----- | ----- | ----- | ----- | -----|
    | cluewsc-dev | 5ab83b | accuracy | gen | 96.86 |
    | cluewsc-test | 5ab83b | accuracy | gen | 92.83 |

    | dataset | version | metric | mode | deepseek-v3-h800 |
    |----- | ----- | ----- | ----- | -----|
    | cluewsc-dev | 5ab83b | accuracy | gen | 96.23 |
    | cluewsc-test | 5ab83b | accuracy | gen | 92.93 |
    ```
