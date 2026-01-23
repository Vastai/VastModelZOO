# StableLM

## Model Arch

## Model Info
### Support Models

| models  | tips |
| :---: | :--: |
| [stablelm-2-1_6b](https://huggingface.co/stabilityai) |[modeling_stablelm_vacc.py](./source_code/modeling_stablelm_vacc.py) |
| [stablelm-2-1_6b-chat](https://huggingface.co/stabilityai) |[modeling_stablelm_vacc.py](./source_code/modeling_stablelm_vacc.py) |
| [stablelm-2-12b](https://huggingface.co/stabilityai) |[modeling_stablelm_vacc.py](./source_code/modeling_stablelm_vacc.py) |
| [stablelm-2-12b-chat](https://huggingface.co/stabilityai) |[modeling_stablelm_vacc.py](./source_code/modeling_stablelm_vacc.py) |


## Build_In Deploy

### step.1 模型准备
#### stablelm
1. 参考`Support Models`列表下载模型权重
2. 为了方便部署`stablelm`系列模型， 在官方源码的基础上， 对`modeling_stablelm.py`做一些修改，其中左图为修改的代码
- [modeling_stablelm_vacc.py](./source_code/modeling_stablelm_vacc.py)
    - 去掉flash_attention, 修改相关依赖的导入方式

    ![](../../images/llm/stablelm/modify.png)


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
    - [hf_stablelm_fp16.yaml](./build_in/build/hf_stablelm_fp16.yaml)
    - [hf_stablelm_int8.yaml](./build_in/build/hf_stablelm_int8.yaml)

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`

    ```bash
    cd stablelm
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/hf_stablelm_fp16.yaml
    vamc compile ../build_in/build/hf_stablelm_int8.yaml
    ```

### step.4 模型推理
1. 参考大模型部署推理工具：[vastgenx](../../docs/vastgenx/README.md)

### Tips
- **LLM模型请先查看概要指引**，[Tips🔔](../README.md)
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
