# TigerBot

- [TigerBot: An Open Multilingual Multitask LLM](https://arxiv.org/abs/2312.08688)


## Model Arch

## Model Info
### Support Models

| models  | tips |
| :---: | :--: |
| [tigerbot-7b-chat](https://huggingface.co/TigerResearch/) | - |
| [tigerbot-13b-chat-v5](https://huggingface.co/TigerResearch/) | - |


## Build_In Deploy

### step.1 模型准备
1. 参考`Support Models`列表下载模型权重

### step.2 数据集
1. 量化校准数据集：
    - [allenai/c4](https://hf-mirror.com/datasets/allenai/c4/tree/main/en)
        - c4-train.00000-of-01024.json.gz
        - c4-validation.00000-of-00008.json.gz
    - [ceval/ceval-exam](https://hf-mirror.com/datasets/ceval/ceval-exam/tree/main)
        - ceval-exam.zip
    - [yahma/alpaca-cleaned](https://hf-mirror.com/datasets/yahma/alpaca-cleaned/tree/main)
        - alpaca_data_cleaned.json

### step.3 模型转换
1. 根据具体模型修改模型转换配置文件
    - [hf_tigerbot_fp16.yaml](./build_in/build/hf_tigerbot_fp16.yaml)
    - [hf_tigerbot_int8.yaml](./build_in/build/hf_tigerbot_int8.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`

    ```bash
    cd tigerbot
    mkdir workspace
    cd workspace
    vamc compile ./build_in/build/hf_tigerbot_fp16.yaml
    vamc compile ./build_in/build/hf_tigerbot_int8.yaml
    ```

### step.4 模型推理
1. 参考大模型部署推理工具：[vastgenx](../../docs/vastgenx/README.md)

### Tips
- **LLM模型请先查看概要指引**，[Tips🔔](../README.md)
- tigerbot_7b
    - 使用`LlamaDynamicNTKScalingRotaryEmbedding`， 在seq_len不大于2048（max_position_embeddings）时与`LlamaRotaryEmbedding`无差异
    - 使用`pretraining_tp=4`, 将QKV线性层进行切分(其它线性层也会切分)，推理效果等同于`pretraining_tp=1`, 在编译vacc模型时需要将config.json中的pretraining_tp修改为1， 否则无法导出正确的三件套
- tigerbot_13b
    - 该模型vocab_size大小非16整数倍， 需要做vacab_size padding， 因此需要`vamc-2.3.4`及以上版本进行转换
- 依赖配置
    ```bash
    protobuf==3.20.3
    torch==2.1.0
    onnx==1.14.0
    onnxsim==0.4.35
    onnxruntime==1.13.1
    accelerate==0.25.0
    transformers>=4.31.0
    ```
