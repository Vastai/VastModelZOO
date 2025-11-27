# WizardLM

- [WizardLM](https://hf-mirror.com/dreamgen/WizardLM-2-7B)


## Model Arch
- 基于`mistralai/Mistral-7B-v0.1`微调
- 该模型利用了分组查询注意力(GQA)和滑动窗口注意力(SWA)的机制，提高了推理速度和效率。GQA加速了推理速度，减少了解码过程中的内存需求，从而实现更高的批处理大小和吞吐量；SWA通过降低计算成本，更有效地处理任意长度的序列。
- 结构参考llama2：[vastml](http://10.23.4.211:8001/llm/llama/)

## Model Info
### Support Models

| models  | tips |
| :---: | :--: |
| [WizardLM-2-7B](https://hf-mirror.com/dreamgen/WizardLM-2-7B) |GQA，Base on [mistral](../mistral/README.md) |

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
    - 参考：[mistral](../mistral/README.md)
    - [hf_wizardlm_fp16.yaml](./build_in/build/hf_wizardlm_fp16.yaml)
    - [hf_wizardlm_int8.yaml](./build_in/build/hf_wizardlm_int8.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`

    ```bash
    cd wizardlm
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/hf_wizardlm_fp16.yaml
    vamc compile ../build_in/build/hf_wizardlm_int8.yaml
    ```
### step.4 模型推理
1. 参考大模型部署推理工具：[vastgenx](../../docs/vastgenx/README.md)


### Tips
- **LLM模型请先查看概要指引**，[Tips🔔](../README.md)
- GQA模型，vamc2x版本需配置enable_kv_share编译参数，在vamc3x以后无需显式设置，在工具内判断是否为GQA模型，自动添加
