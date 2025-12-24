# Build_In Deploy

## step.1 模型准备

1. 下载模型权重

- Embedding

| Model Type       | Models               | Size | Layers | Sequence Length | Embedding Dimension | MRL Support | Instruction Aware |
|------------------|----------------------|------|--------|-----------------|---------------------|-------------|----------------|
| Text Embedding   | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 0.6B | 28     | 32K             | 1024                | Yes         | Yes            |


- ReRanker

| Model Type       | Models               | Size | Layers | Sequence Length | Embedding Dimension | MRL Support | Instruction Aware |
|------------------|----------------------|------|--------|-----------------|---------------------|-------------|----------------|
| Text Reranking   | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 0.6B | 28     | 32K             | -                   | -           | Yes            |
 
> **Note**:
> - `MRL Support` 表示嵌入模型是否支持自定义最终嵌入的维度。 
> - `Instruction Aware` 标注了嵌入或重排序模型是否支持根据不同任务定制输入指令。


2. 模型修改

- Embedding
    - 为在瀚博软件栈部署`Qwen3`系列模型，在官方源码的基础上，需要对`modeling_qwen2.py`做一些修改，其中左图为修改的代码

    - [modeling_qwen2_vacc.py](../source_code/embedding/modeling_qwen2_vacc.py)
        - 修改相关依赖的导入方式
        ![](../../../../images/nlp/qwen3-0.6b/Snipaste_1.png)
        - 基于config.insert_slice来判断是否插入strided_slice
        ![](../../../../images/nlp/qwen3-0.6b/Snipaste_2.png)
        - 删除lm_head，因为Embedding模型不需要lm_head
        ![](../../../../images/nlp/qwen3-0.6b/Snipaste_3.png)
        ![](../../../../images/nlp/qwen3-0.6b/Snipaste_4.png)
        
    - [configuration_qwen2_vacc.py](../source_code/embedding/configuration_qwen2_vacc.py)
        - 修改对于相关依赖的导入方式
        ![](../../../../images/nlp/qwen3-0.6b/Snipaste_5.png)

    - [config_vacc.json](../source_code/embedding/config_vacc.json)
        - 添加auto_map选项
        - 修改use_cache为false
        - 添加_attn_implementation选项，并将其只配置为eager；
        
        ![](../../../../images/nlp/qwen3-0.6b/Snipaste_6.png)
    
    - 将以上修改后文件，放置于原始权重目录下

- Reranker
    - [modeling_qwen2_vacc.py](../source_code/reranker/modeling_qwen2_vacc.py)
        - 修改相关依赖的导入方式
        ![](../../../../images/nlp/qwen3-0.6b/Snipaste_1.png)
        - 基于config.insert_slice来判断是否插入strided_slice
        ![](../../../../images/nlp/qwen3-0.6b/Snipaste_2.png)

    - [config_vacc.json](../source_code/reranker/config_vacc.json)
        
        - 添加auto_map选项
        - 添加_attn_implementation选项，并将其只配置为eager；
        ![](../../../../images/nlp/qwen3-0.6b/Snipaste_7.png)
    
    - [configuration_qwen2_vacc.py](../source_code/embedding/configuration_qwen2_vacc.py)
        - 同Embedding模型
    
    - 将以上修改后文件，放置于原始权重目录下

## step.2 数据集

1. 精度评估数据集：
    - embedding
        - 英文：[mteb/sts12-sts](https://huggingface.co/datasets/mteb/sts12-sts)
        - 中文：[C-MTEB/BQ](https://huggingface.co/datasets/C-MTEB/BQ)
    - reranker：[zyznull/msmarco-passage-ranking](https://huggingface.co/datasets/zyznull/msmarco-passage-ranking)
    - 数据集下载和转换为jsonl格式：[download_datasets.py](../../common/source_code/download_datasets.py)

## step.3 模型转换
1. 根据具体模型修改模型转换配置文件
    - [embedding_config_fp16.yaml](./build/embedding_config_fp16.yaml)
    - [reranker_config_fp16.yaml](./build/reranker_config_fp16.yaml)

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

    > - 配置文件中的input_ids需根据需求进行修改，例如：编译512尺寸的模型，则为input_ids: [[512],[]]；编译1024尺寸的模型，则为input_ids: [[1024],[]]

2. 模型编译
    ```bash
    cd qwen3-0.6b
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/embedding_config_fp16.yaml
    vamc compile ../build_in/build/reranker_config_fp16.yaml
    ```

## step.4 模型推理
1. 推理：[demo.py](./vsx/demo.py)
    - 配置模型路径等参数，推理脚本内指定的文本对

    ```bash
    #embedding
    python ../build_in/vsx/demo.py \
        --vacc_weight /path/to/vacc_deploy/Qwen3-Embedding-0.6B-VACC-512/prefill_512_rank0/mod \
        --torch_weight /path/to/Qwen3-Embedding-0.6B \
        --task embedding \
        --eval_engine vacc \
        --seqlen 512 \
        --batch_size 1 \
        --device_id 0
    
    #reranker
    python ../build_in/vsx/demo.py \
        --vacc_weight /path/to/vacc_deploy/Qwen3-Reranker-0.6B-VACC-512/prefill_512_rank0/mod \
        --torch_weight /path/to/Qwen3-Reranker-0.6B \
        --task reranker \
        --eval_engine vacc \
        --seqlen 512 \
        --batch_size 1 \
        --device_id 0
    ```

## step.5 性能精度测试
1. 性能测试
    - 参考推理脚本：[performace.py](./vsx/performace.py)，修改参数并运行如下脚本

    ```bash
    #测试最大吞吐
    python3 ../build_in/vsx/performace.py \
        --model_prefix /path/to/vacc_deploy/Qwen3-Embedding-0.6B-VACC-512/prefill_512_rank0/mod \
        --device_ids [0] \
        --batch_size 1 \
        --instance 1 \
        --iterations 100 \
        --percentiles "[50, 90, 95, 99]" \
        --input_host 1 \
        --queue_size 1 

    #测试最小时延
    python3 ../build_in/vsx/performace.py \
        --model_prefix /path/to/vacc_deploy/Qwen3-Embedding-0.6B-VACC-512/prefill_512_rank0/mod \
        --device_ids [40] \
        --batch_size 1 \
        --instance 1 \
        --iterations 100 \
        --percentiles "[50, 90, 95, 99]" \
        --input_host 1 \
        --queue_size 0
    ```

2. 精度测试：[demo.py](./vsx/demo.py)
    - 配置模型路径等参数，指定`--eval_mode`参数为True，进行精度评估

    ```bash
    #embedding
    python ../build_in/vsx/demo.py \
        --vacc_weight /path/to/Qwen3-Embedding-0.6B-VACC-512/prefill_512_rank0/mod \
        --torch_weight /path/to/Qwen3-Embedding-0.6B \
        --task embedding \
        --eval_mode \
        --eval_engine vacc \
        --eval_dataset /path/to/mteb-sts12-sts_test.jsonl \
        --seqlen 512 \
        --batch_size 1 \
        --device_id 0

    #reranker
    python ../build_in/vsx/demo.py \
        --vacc_weight /path/to/Qwen3-Reranker-0.6B-VACC-512/prefill_512_rank0/mod \
        --torch_weight /path/to/Qwen3-Reranker-0.6B \
        --task reranker \
        --eval_mode \
        --eval_engine vacc \
        --eval_dataset /path/to/zyznull-msmarco-passage-ranking_dev.jsonl \
        --seqlen 512 \
        --batch_size 1 \
        --device_id 0
    ```
    
## Tips
- 目前仅支持fp16精度，后续会支持int8精度