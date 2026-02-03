# 概述

本文档旨在指导用户如何基于 vLLM 在瀚博硬件设备上部署 Qwen3 Embedding 系列模型，以及测试模型的精度。

## 模型支持

- Embedding

| Model Type       | Models               | Size | Layers | Sequence Length | Embedding Dimension | MRL Support | Instruction Aware |
|------------------|----------------------|------|--------|-----------------|---------------------|-------------|----------------|
| Text Embedding   | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 0.6B | 28     | 32K             | 1024                | Yes         | Yes            |
| Text Embedding   | [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | 4B | 36     | 32K             | 2560                | Yes         | Yes            |
| Text Embedding   | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 8B | 36     | 32K             | 4096                | Yes         | Yes            |
    

- ReRanker

| Model Type       | Models               | Size | Layers | Sequence Length | Embedding Dimension | MRL Support | Instruction Aware |
|------------------|----------------------|------|--------|-----------------|---------------------|-------------|----------------|
| Text Reranking   | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 0.6B | 28     | 32K             | -                   | -           | Yes            |
| Text Reranking   | [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | 4B | 36     | 32K             | -                   | -           | Yes            |
| Text Reranking   | [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | 8B | 36     | 32K             | -                   | -           | Yes            |

> **Note**:
> - `MRL Support` 表示嵌入模型是否支持自定义最终嵌入的维度。 
> - `Instruction Aware` 标注了嵌入或重排序模型是否支持根据不同任务定制输入指令。

## 使用限制

- Embedding

  | model | parallel | seq limit | tips|
  |:--- |:--- | :-- | :-- | 
  | Qwen3-Embedding-0.6B | TP1/2/4/8 | max-model-len 32k | max-concurrency 4|
  | Qwen3-Embedding-4B | TP1/2/4/8 | max-model-len 32k | max-concurrency 4|
  | Qwen3-Embedding-8B | TP2/4/8 | max-model-len 32k | max-concurrency 4|

- ReRanker

  | model | parallel | seq limit | tips|
  |:--- |:--- | :-- | :-- |
  | Qwen3-Reranker-0.6B | TP1/2/4/8 | max-model-len 32k | max-concurrency 4|
  | Qwen3-Reranker-4B | TP1/2/4/8 | max-model-len 32k |max-concurrency 4|
  | Qwen3-Reranker-8B | TP2/4/8 | max-model-len 32k | max-concurrency 4|



## 模型下载
1. 通过hf-mirror下载

- 参考[hf-mirror](https://hf-mirror.com/)下载权重
  ```shell
  wget https://hf-mirror.com/hfd/hfd.sh
  chmod a+x hfd.sh
  export HF_ENDPOINT=https://hf-mirror.com
  apt install aria2
  ./hfd.sh Qwen/Qwen3-Embedding-0.6B -x 10 --local-dir Qwen3-Embedding-0.6B
  ```

2. 或通过modelscope下载

- 参考[modelscope](https://modelscope.cn/docs/models/download)下载权重
  ```shell
  pip install modelscope -i https://mirrors.ustc.edu.cn/pypi/web/simple
  export PATH=$PATH:~/.local/bin
  modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir ./Qwen3-Embedding-0.6B
  ```


## 启动模型服务

1. 参考官方启动命令：[vllm](https://docs.vllm.ai/en/latest/cli/#serve)

  ```bash
  #Embedding模型服务
  docker run \
      -e VACC_VISIBLE_DEVICES=0 \
      --privileged=true --shm-size=256g \
      --name vllm_service_embedding \
      -v /path/to/model:/weights/ \
      -p 8000:8000 \
      --ipc=host \
      harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-25.12.SP1 \
      vllm serve /weights/Qwen3-Embedding-0.6B \
      --trust-remote-code \
      --enforce-eager \
      --host 0.0.0.0 \
      --port 8000 \
      --task embed \
      --served-model-name Qwen3-Embedding-0.6B
  ```

  ```bash
  #ReRanker模型服务
  docker run \
      -e VACC_VISIBLE_DEVICES=1 \
      --privileged=true --shm-size=256g \
      --name vllm_service_reranker \
      -v /path/to/model:/weights/ \
      -p 8001:8001 \
      --ipc=host \
      harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-25.12.SP1 \
      vllm serve /weights/Qwen3-Reranker-0.6B \
      --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
      --enforce-eager \
      --host 0.0.0.0 \
      --port 8001 \
      --task score  \
      --served-model-name Qwen3-Reranker-0.6B
  ```

- 参数说明如下
  - `VACC_VISIBLE_DEVICES`: 指定使用哪一个GPU设备。

  - `-v`：原始模型权重所在路径。请根据实际情况替换。

  - `--port`：模型服务端口。

  - `--served-model-name`：模型名称。

  - `--task`：模型任务类型，`embed`表示Embedding模型，`score`表示ReRanker模型。

  - `--hf_overrides`：在加载模型时动态修改模型配置，`Qwen3-Reranker-0.6B`模型需要设置`--hf_overrides`参数。

  - `harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-25.12.SP1`：vllm_vacc镜像地址，请根据实际情况替换, 具体版本可从首页[依赖软件](../../../../README.md)中获取。


## 模型精度测试

1. 通过EvalScope的RAGEval后端进行模型精度测试，evalscope安装参考：[installation](https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html) 评测后端RAGEval安装参考：[RAGEval](https://evalscope.readthedocs.io/zh-cn/v1.3.0/user_guides/backend/rageval_backend/mteb.html#id3)

    > 注意：为适配模型API服务，需进行适当修改evalscope源码，详见：[modify_detail.md](./modify_detail.md)


2. 启动 vLLM 模型服务

3. 精度测试
- Embedding模型
参考脚本：[precision_embedding.py](../../../../tools/evalscope/precision_embedding.py)，配置测评数据集及评测参数等信息，执行脚本获取精度测评结果

    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    #embedding模型测试
    python ../../../../tools/evalscope/precision_embedding.py
    ```

- ReRanker模型
参考脚本：[precision_reranker.py](../../../../tools/evalscope/precision_reranker.py)，配置测评数据集及评测参数等信息，执行脚本获取精度测评结果

    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    #reranker模型测试
    python ../../../../tools/evalscope/precision_reranker.py
    ```

  - 测评主要参数：
    - work_dir：评测结果保存路径。

    - eval_backend：评测后端，设置为`RAGEval`,表示使用 RAGEval 评测后端。

    - eval_config：评测配置。

      - tool：评测工具，使用`MTEB`。

      - model：模型配置列表。

        - model_name：`str` 模型名称，该参数设置应与模型服务启动脚本中`--served-model-name`参数一致。

        - api_base：`str` 模型API服务地址。

        - api_key：`str` 模型API密钥。默认值：Empty。

        - dimensions：`int` 模型输出维度。

        - encode_kwargs：`dict` 编码的关键字参数。

          - batch_size：`int`编码批次大小。

        - is_cross_encoder：`bool` 模型是否为交叉编码器，默认为 False；reranker模型需设置为True。

        - model_kwargs: `dict` 模型的关键字参数。

          - instruction_template：`str` 指令模板。

          - instruction_dict_path：`str` 指令字典路径。
          
          - task_name：`list` 评测任务列表,仅测试reranker模型需要设置。

          - embed_results：`list` embedding模型结果列表,仅测试reranker模型需要设置。

      - eval：评测参数。

        - tasks：`List[str]` 评测任务列表。

        - top_k：`int`选取前 K 个结果，测试reranker模型需要设置。

        - verbosity：`int`详细程度，范围为 0-3 。

        - output_folder：`str` 评测结果保存路径。

        - overwrite_results：`bool`是否覆盖结果，默认为 True 。