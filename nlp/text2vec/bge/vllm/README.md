# 概述

本文档旨在指导用户如何基于 vLLM 在瀚博硬件设备上部署 BGE 系列模型，以及测试模型的精度。

## 模型支持

- Embedding
    | Model  | Dimension | Sequence Length | Language |
    | :------: | :-------: | :-------------: | :------: |
    |      [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)       |   1024    |      8192       | multilingual; unified fine-tuning (dense, sparse, and colbert) </br> from bge-m3-unsupervised |
    | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |    384    |       512       |                      English model                         |
    | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |    768    |       512       |                        English model                         |
    | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |   1024    |       512       |                      English model                         |
    | [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) |    384    |       512       |                      Chinese model
    | [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) |    768    |       512       |                        Chinese model                         |
    | [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) |   1024    |       512       |                      Chinese model                         |
    
    >  base on XLMRobertaModel

- ReRanker

    | Model |   Base Model  |  Dimension | Sequence Length  | Note |Language |
    | :------: | :------: | :------: | :--: |:--: |:--: | 
    | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |         [bge-m3](https://huggingface.co/BAAI/bge-m3)  |  1024 | 8192 | 轻量级重排序模型，具有强大的多语言能力，易于部署，推理速度快。 | 多种语言 | 
    | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)  | 768 | 512 | 轻量级重排序模型，易于部署，推理速度快。 | 中英文  |  
    | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | [xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) |  1024 | 512 | 轻量级重排序模型，易于部署，推理速度快。 |  中英文  |

## 使用限制

- Embedding

  | model | parallel | seq limit | tips|
  |:--- |:--- | :-- | :-- | 
  | bge-m3 | TP1/2/4/8 | max-model-len 8192 | max-concurrency 4|
  | bge-small-* | TP1/2/4/8 | max-model-len 512 | max-concurrency 4|
  | bge-base-* | TP1/2/4/8 | max-model-len 512 | max-concurrency 4|
  | bge-large-* | TP1/2/4/8 | max-model-len 512 | max-concurrency 4|

- ReRanker

  | model | parallel | seq limit | tips|
  |:--- |:--- | :-- | :-- | 
  | bge-reranker-v2-m3 | TP1/2/4/8 | max-model-len 8192 | max-concurrency 4|
  | bge-reranker-base | TP1/2/4/8 | max-model-len 512 | max-concurrency 4|
  | bge-reranker-large | TP1/2/4/8 | max-model-len 512 | max-concurrency 4|


## 模型下载
1. 通过hf-mirror下载

- 参考[hf-mirror](https://hf-mirror.com/)下载权重
  ```shell
  wget https://hf-mirror.com/hfd/hfd.sh
  chmod a+x hfd.sh
  export HF_ENDPOINT=https://hf-mirror.com
  apt install aria2
  ./hfd.sh BAAI/bge-m3 -x 10 --local-dir bge-m3
  ```

2. 或通过modelscope下载

- 参考[modelscope](https://modelscope.cn/docs/models/download)下载权重
  ```shell
  pip install modelscope -i https://mirrors.ustc.edu.cn/pypi/web/simple
  export PATH=$PATH:~/.local/bin
  modelscope download --model BAAI/bge-m3 --local_dir ./bge-m3
  ```


## 启动模型服务

1. 参考官方启动命令：[vllm](https://docs.vllm.ai/en/latest/cli/#serve)

  ```bash
  docker run \
      -e VACC_VISIBLE_DEVICES=0 \
      --privileged=true --shm-size=256g \
      --name vllm_service \
      -v /path/to/model:/weights/ \
      -p 8000:8000 \
      --ipc=host \
      harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-25.12.SP1 \
      vllm serve /weights/bge-m3 \
      --tensor-parallel-size 1 \
      --enforce-eager \
      --host 0.0.0.0 \
      --port 8000 \
      --served-model-name bge-m3

  ```

- 参数说明如下
  - `VACC_VISIBLE_DEVICES`: 指定使用哪一个GPU设备。

  - `--tensor-parallel-size`：张量并行数。

  - `-v`：原始模型权重所在路径。请根据实际情况替换。

  - `--port`：模型服务端口。

  - `--served-model-name`：模型名称。

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