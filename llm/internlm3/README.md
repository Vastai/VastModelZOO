# Internlm3

- [InternLM2 Technical Report](https://arxiv.org/abs/2403.17297)

## Model Arch
![](../../images/llm/internlm/internlm2_arch.png)

## Model Info
### Support Models

| models  | tips |
| :---: | :--: |
| [internlm/internlm3-8b-instruct](https://huggingface.co/internlm/internlm3-8b-instruct) |  GQA，[modeling_internlm3_vacc.py](./build_in/source_code/modeling_internlm3_vacc.py) |


## Build_In Deploy

### step.1 模型准备

#### internlm3
1. 参考`Support Models`列表下载模型权重
2. 为了方便部署`Internlm3`系列模型， 在官方源码的基础上， 对`modeling_internlm3.py`做一些修改，其中左图为修改的代码
- [modeling_internlm3_vacc.py](./source_code/modeling_internlm3_vacc.py)
    - 参考llama系列，修改InternLM3RotaryEmbedding

    ![](../../images/llm/internlm/Snipaste_2025-02-27_10-50-48.png)
    ![](../../images/internlm/Snipaste_2025-02-27_10-54-11.png)
    - 参考llama系列，使用eager注意力方式，修改InternLM3Attention
    ![](../../images/llm/internlm/Snipaste_2025-02-27_10-55-15.png)
    - 修改attention_mask的生成方式；并在transformers==4.45下不使用position_embeddings
    ![](../../images/llm/internlm/Snipaste_2025-02-27_10-59-10.png)
    - 修改transformers==4.45版本下的get_seq_length方法实现方式
    ![](../../images/llm/internlm/Snipaste_2025-02-27_11-02-57.png)
    - 其它微小改动，请直接与对比原始modeling查阅
    

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
    cd internlm3
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/hf_internlm_fp16.yaml
    vamc compile ../build_in/build/hf_internlm_int8.yaml
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
    transformers==4.45.0
    ```

