# Yi

- [Yi: Open Foundation Models by 01.AI](https://arxiv.org/abs/2403.04652)

## Model Arch
- 基于LLAMA2
- Yi-6B, Yi-9B, Yi-34B均使用GQA


## Model Info
### Support Models

| models  | tips |
| :---: | :--: |
| [Yi-6B](https://huggingface.co/01-ai) |GQA |
| [Yi-9B](https://huggingface.co/01-ai) |GQA |
| [Yi-34B](https://huggingface.co/01-ai) |GQA |


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
    - [hf_yi_fp16.yaml](./build_in/build/hf_yi_fp16.yaml)
    - [hf_yi_int8.yaml](./build_in/build/hf_yi_int8.yaml)

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`

    ```bash
    cd yi
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/hf_yi_fp16.yaml
    vamc compile ../build_in/build/hf_yi_int8.yaml
    ```

### step.4 模型推理
1. 参考大模型部署推理工具：[vastgenx](../../docs/vastgenx/README.md)

### Tips
- **LLM模型请先查看概要指引**，[Tips🔔](../README.md)
- GQA模型，vamc2x版本需配置enable_kv_share编译参数，在vamc3x以后无需显式设置，在工具内判断是否为GQA模型，自动添加
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
