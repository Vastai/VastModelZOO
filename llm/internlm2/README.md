# Internlm2

- [InternLM2 Technical Report](https://arxiv.org/abs/2403.17297)

## Model Arch
![](../../images/llm/internlm/internlm2_arch.png)

## Model Info
### Support Models

| models  | tips |
| :---: | :--: |
| [internlm/internlm2-1_8b](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c) | GQA，[modeling_internlm2_vacc.py](./build_in/source_code/modeling_internlm2_vacc.py) |
| [internlm/internlm2-chat-1_8b](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c) |  GQA，[modeling_internlm2_vacc.py](./build_in/source_code/modeling_internlm2_vacc.py) |
| [internlm/internlm2-7b](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c) | GQA，[modeling_internlm2_vacc.py](./build_in/source_code/modeling_internlm2_vacc.py) |
| [internlm/internlm2-chat-7b](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c) |  GQA，[modeling_internlm2_vacc.py](./build_in/source_code/modeling_internlm2_vacc.py) |
| [internlm/internlm2-20b](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c) |  GQA，[modeling_internlm2_vacc.py](./build_in/source_code/modeling_internlm2_vacc.py) |
| [internlm/internlm2-chat-20b](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c) | GQA，[modeling_internlm2_vacc.py](./build_in/source_code/modeling_internlm2_vacc.py) |


## TVM_VACC部署

### step.1 模型准备

#### internlm2
1. 参考`Support Models`列表下载模型权重
2. 为了方便部署`Internlm2`系列模型， 在官方源码的基础上， 对`modeling_internlm2.py`做一些修改，其中左图为修改的代码
- [modeling_internlm2_vacc.py](./source_code/modeling_internlm2_vacc.py)
    - 参考glm2拆分qkv, 方便后续的tensor并行

    ![](../../images/llm/internlm/internlm2_selfattn_forward.png)

    > 其他修改可忽略


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
    - v1/v2/v3模型，编译配置一致
    - [hf_internlm_fp16.yaml](./build_in/build/hf_internlm_fp16.yaml)
    - [hf_internlm_int8.yaml](./build_in/build/hf_internlm_int8.yaml)

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`
    
    ```bash
    vamc compile ./build_in/build/hf_internlm_fp16.yaml
    vamc compile ./build_in/build/hf_internlm_int8.yaml
    ```

### step.4 模型推理
1. 参考大模型部署推理工具：[vastgenx: v1.1.0+](../../docs/vastgenx/README.md)

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
    transformers>=4.31.0 # 推荐4.40
    ```

