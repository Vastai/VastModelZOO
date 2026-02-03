# OpenBuddy

- [OpenBuddy](https://github.com/OpenBuddy/OpenBuddy/blob/main/README.zh.md)


## Model Arch
- 基于qwen/mistral2等模型构建，OpenBuddy经过微调，包括扩展词汇表、增加常见字符和增强token嵌入，提升模型能力
- 结构参考上述基础模型

## Model Info
### Support Models

| models  | tips |
| :---: | :--: |
| [openbuddy-mistral2-7b-v20.3-32k](https://huggingface.co/collections/OpenBuddy/our-selected-models-65369270912eef259074d3dc) | - |
| [openbuddy-qwen1.5-14b-v21.1-32k](https://huggingface.co/collections/OpenBuddy/our-selected-models-65369270912eef259074d3dc) | - |
| [openbuddy-deepseek-67b-v18.1-4k](https://huggingface.co/collections/OpenBuddy/our-selected-models-65369270912eef259074d3dc) | GQA |


## Build_In Deploy

### step.1 模型准备

1. 参考`Support Models`列表下载模型权重
2. 网络修改
    - openbuddy-mistral2-7b-v20.3-32k，参考[mistral](../mistral/README.md)
    - openbuddy-qwen1.5-14b-v21.1-32k，参考[qwen](../qwen1.5/README.md)
    - openbuddy-deepseek-67b-v18.1-4k，参考[llama](../llama/README.md)


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
    - [hf_openbuddy_fp16.yaml](./build_in/build/hf_openbuddy_fp16.yaml)
    - [hf_openbuddy_int8.yaml](./build_in/build/hf_openbuddy_int8.yaml)

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`

    ```bash
    cd openbuddy
    mkdir workspace
    cd workspace
    vamc compile ./build_in/build/hf_openbuddy_fp16.yaml
    vamc compile ./build_in/build/hf_openbuddy_int8.yaml
    ```

### step.4 模型推理
1. 参考大模型部署推理工具：[vastgenx](../../tools/vastgenx/README.md)


### Tips
- **LLM模型请先查看概要指引**，[Tips🔔](../README.md)
