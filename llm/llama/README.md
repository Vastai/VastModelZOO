# LLaMA

- Technical Report
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- Huggingface
    - https://huggingface.co/meta-llama


## Model Arch

自从OpenAI推出Chat GPT系列后，也标志着自然语言处理技术的一个重要里程碑——大模型LLM（Large Language Model）的爆火。开源社区的先锋当属Meta公司推出的LLAMA(Large Language Model Meta AI)系列，作为Decoder-Only结构的代表，为后续开源大模型指明了方向。

![llama_arch](../../images/llm/llama_arch.png)


### LLaMA v1
- 高质量数据集
    - 预训练数据大约包含 1.4T tokens
    - 数据处理步骤：筛选低质量数据、数据去重、数据多样性
- 归一化优化，增加Pre-Normalization步骤，使用RMSNorm代替LayerNorm，简化计算，提升效率
- 激活函数优化，使用SwiGLU代替ReLU以提高性能
- Rotary Embeddings旋转位置编码，RoPE旋转位置编码的核心思想是“通过绝对位置编码的方式实现相对位置编码”，这一构思具备了绝对位置编码的方便性，同时可以表示不同token之间的相对位置关系


## Build_In Deploy

### step.1 模型准备

1. 下载模型权重

    | models  | tips |
    | :--- | :--: | 
    | [alexl83/LLaMA-33B-HF](https://huggingface.co/alexl83/LLaMA-33B-HF)   | MHA |

    > - 其它基于`llama`微调的模型(`model_type:llama`)，转换及推理测试参考`llama`系列即可
    > - `meta-llama`开源的模型均不支持商用，请查阅原始许可证


### step.2 数据集

1. 量化校准数据集：
    - [allenai/c4](https://hf-mirror.com/datasets/allenai/c4/tree/main/en)
        - c4-train.00000-of-01024.json.gz
        - c4-validation.00000-of-00008.json.gz
    - [ceval/ceval-exam](https://hf-mirror.com/datasets/ceval/ceval-exam/tree/main)
        - ceval-exam.zip
    - [yahma/alpaca-cleaned](https://hf-mirror.com/datasets/yahma/alpaca-cleaned/tree/main)
        - alpaca_data_cleaned.json

2. 性能测试不定长数据集：[ShareGPT_V3_unfiltered_cleaned_split.json](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json)
3. 精度评估数据集：[OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)

### step.3 模型转换

1. 参考瀚博训推软件生态链文档，获取模型转换工具: [vamc v3.0+](../../docs/vastai_software.md)
2. 根据具体模型，修改模型转换配置文件
    - v1/v2/v3模型，编译配置一致
    - [hf_llama_fp16.yaml](./build_in/build/hf_llama_fp16.yaml)
    - [hf_llama_int8.yaml](./build_in/build/hf_llama_int8.yaml)

    ```bash
    vamc compile ./build_in/build/hf_llama_fp16.yaml
    vamc compile ./build_in/build/hf_llama_int8.yaml
    ```


### step.4 模型推理
1. 参考瀚博训推软件生态链文档，获取模型推理工具：[llmdeploy v1.6+](../../docs/vastai_software.md)
2. 参考llmdeploy工具文档，进行模型推理、性能和精度测试

### Tips
- **LLM模型请先查看概要指引**，[Tips🔔](../README.md)
- llama系列，不会对原始llama_modeling.py进行修改，为兼容多版本模型，建议依赖配置如下：
    ```bash
    protobuf==3.20.3
    torch==2.1.0
    onnx==1.14.0
    onnxsim==0.4.28
    onnxruntime==1.13.1
    accelerate==0.25.0
    transformers==4.34.0
    ```
