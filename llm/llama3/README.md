# LLaMA3

- Technical Report
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
    - [Llama 2: Open Foundation and Fine-Tuned Chat Models]( https://arxiv.org/abs/2307.09288)
- Huggingface
    - https://huggingface.co/meta-llama


## Model Arch
![llama_arch](../../images/llm/llama_arch.png)


### LLaMA v3
- 继续提升训练数据的数量和质量
    - v3使用从公开来源收集的超过15T的token，是v2的七倍，其中包含的代码数据则是v2的四倍
    - 超过5%的v3预训练数据集由涵盖30多种语言的高质量非英语数据组成
    - 启发式过滤器、NSFW 筛选器、语义重复数据删除方法和文本分类器来预测数据质量
- 全面优化训练流程，使用数据并行化、模型并行化和管道并行化，训练效率比v2高出3倍，
- 支持8K长文本，改进的tokenizer具有128K token的词汇量，可实现更好的性能
- 同步发布Guard 2、Code Shield和CyberSec Eval 2的新版信任和安全工具


## Build_In Deploy

### step.1 模型准备

1. 下载模型权重

    | models  | tips |
    | :--- | :--: | 
    | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B/) | GQA |
    | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/) | GQA |
    | [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B/) | GQA |
    | [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/) | GQA |
    | [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/) | GQA |
    | [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/) | GQA |
    | [meta-llama/Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B/) | GQA |
    | [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct/) | GQA |
    | [meta-llama/Meta-Llama-3.2-1B](https://huggingface.co/meta-llama/Meta-Llama-3.2-1B/) | GQA |
    | [meta-llama/Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.2-1B-Instruct/) | GQA |
    | [meta-llama/Meta-Llama-3.2-3B](https://huggingface.co/meta-llama/Meta-Llama-3.2-3B/) | GQA |
    | [meta-llama/Meta-Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.2-3B-Instruct/) | GQA |
    | [meta-llama/Meta-Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.3-70B-Instruct/) | GQA |

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
    - [hf_llama3_fp16.yaml](./build_in/build/hf_llama3_fp16.yaml)
    - [hf_llama3_int8.yaml](./build_in/build/hf_llama3_int8.yaml)

    ```bash
    vamc compile ./build_in/build/hf_llama3_fp16.yaml
    vamc compile ./build_in/build/hf_llama3_int8.yaml
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

## Pytorch Deploy

### step.1 模型准备
|  models |    demo_code    |  tips |
| :------ | :------: | :------: |
|[meta-llama/Llama-3.1-8B-Instruct](https://hf-mirror.com/meta-llama/Llama-3.1-8B-Instruct)|[demo](./pytorch/demo/llama3.1_8b.py) |  - |

### step.2 模型推理
- 基于`torch_vacc`在`VA16`硬件下推理，一般基于官方demo进行适当修改，参见上表`demo_code`部分
