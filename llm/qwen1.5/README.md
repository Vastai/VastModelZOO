# Qwen

- Technical Report
    - [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
    - [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
    - [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- Huggingface
    - https://huggingface.co/Qwen


## Model Arch

![qwen_arch](../../images/llm/qwen/qwen_arch.png)

### Qwen v1.5
- 训练数据：使用了高达3万亿个token的数据进行预训练，数据涵盖多个类型、领域和任务，不仅包括基本的语言能力，还包括算术、编码和逻辑推理等高级技能。同时使用了复杂的流程进行数据清洗和质量控制。
    - 文本数据抽取
    - 语言识别
    - 去重
    - 质量控制
    - 安全控制
    - 长序列建模
- 模型结构，基于LLaMA架构
    - embedding和输出映射不进行权重共享，从而达到以内存成本为代价换取获得更好的性能
    - 使用了RoPE(旋转位置编码)进行位置编码。为了优先考虑模型性能并获得更高的精确度，使用FP32精确度的逆频率矩阵，而不是 BF16 或 FP16
    - 在大多数层中移除了Bias，但在QKV层保留以提升模型的外推能力
    - 使用了预归一化(Pre-Norm)和RMSNorm进行规范化。Pre-Norm是使用最广泛的方法，与post-normalization相比，它已被证明能提高训练的稳定性。最近的研究提出了提高训练稳定性的其他方法，官方表示会在模型的未来版本中进行探索。此外，还用 RMSNorm 替代传统的层归一化技术。这一改变在不损害性能的同时提高了效率
    - 使用了SwiGLU作为激活函数。它是Swish和门控线性单元GLU的组合。初步实验表明，基于GLU的激活函数普遍优于其他基线选项，如GeLU
- 长文本外推能力
    - NTK感知插值(NTK-aware interpolation)，无需训练的技术可以调整比例参数以防止在扩展长度时丢失高频信息
    - 动态NTK感知插值(dynamic NTK-aware interpolation)，这是NTK感知插值的改进版本，可以以块为单位动态改变比例参数,避免性能大幅下降
    - LogN-Scaling，根据上下文长度与训练长度的比值，对Q和V的点积进行重新缩放，确保注意力值的熵随着上下文长度的增长而保持稳定
    - 使用分层窗口Self-Attention，将注意力限制在一个上下文窗口内，防止模型关注到太远的内容。并在不同层采用不同的窗口大小，较低的层使用较短的窗口，而较高的层使用较长的窗口
- 注意力模块采用Flash Attention技术，以提高计算效率并减少内存使用
- 使用BFloat16混合精度加速训练
- Base-SFT-RLHF（RM-PPO）训练策略优化


## Build_In Deploy

### step.1 模型准备

1. 下载模型权重

    | models | tips |
    | :---   | :--: |
    | [Qwen/Qwen1.5-0.5B](https://hf-mirror.com/Qwen/Qwen1.5-0.5B) |  MHA |
    | [Qwen/Qwen1.5-0.5B-Chat](https://hf-mirror.com/Qwen/Qwen1.5-0.5B-Chat) |  MHA |
    | [Qwen/Qwen1.5-1.8B](https://hf-mirror.com/Qwen/Qwen1.5-1.8B) |  MHA |
    | [Qwen/Qwen1.5-1.8B-Chat](https://hf-mirror.com/Qwen/Qwen1.5-1.8B-Chat) |  MHA |
    | [Qwen/Qwen1.5-4B](https://hf-mirror.com/Qwen/Qwen1.5-4B) |  MHA |
    | [Qwen/Qwen1.5-4B-Chat](https://hf-mirror.com/Qwen/Qwen1.5-4B-Chat) |  MHA |
    | [Qwen/Qwen1.5-7B](https://hf-mirror.com/Qwen/Qwen1.5-7B) |  MHA |
    | [Qwen/Qwen1.5-7B-Chat](https://hf-mirror.com/Qwen/Qwen1.5-7B-Chat) |  MHA |
    | [Qwen/Qwen1.5-14B](https://hf-mirror.com/Qwen/Qwen1.5-14B) |  MHA |
    | [Qwen/Qwen1.5-14B-Chat](https://hf-mirror.com/Qwen/Qwen1.5-14B-Chat) |  MHA |
    | [Qwen/Qwen1.5-32B](https://hf-mirror.com/Qwen/Qwen1.5-32B) |  GQA |
    | [Qwen/Qwen1.5-32B-Chat](https://hf-mirror.com/Qwen/Qwen1.5-32B-Chat) |  GQA |
    | [Qwen/Qwen1.5-72B](https://hf-mirror.com/Qwen/Qwen1.5-72B) |  MHA |
    | [Qwen/Qwen1.5-72B-Chat](https://hf-mirror.com/Qwen/Qwen1.5-72B-Chat) |  MHA |
    | [Qwen/Qwen1.5-110B-Chat](https://hf-mirror.com/Qwen/Qwen1.5-110B-Chat) |  MHA |

    > - 其它基于Qwen1.5/Qwen2/2.5微调(均为`model_type:qwen2`)，其模型转换及推理测试参考Qwen2系列即可


2. 模型修改
    - 为在瀚博软件栈部署`Qwen1.5`系列模型，在官方源码的基础上，需要对`modeling_qwen2.py`做一些修改，其中左图为修改的代码
    - [modeling_qwen2_vacc.py](./build_in/source_code/modeling_qwen2_vacc.py)
        - 修改相关依赖的导入方式
        ![](../../images/llm/qwen/Snipaste_2024-04-11_14-10-36.png)
        - 基于config.insert_slice来判断是否插入strided_slice
        ![](../../images/llm/qwen/Snipaste_2024-04-15_17-26-31.png)
        - class Qwen2ForCausalLM添加quantize方法，支持per_channel int8量化，[quantization_vacc.py](./build_in/source_code/quantization_vacc.py)
        ![](../../images/llm/qwen/Snipaste_2024-04-15_17-29-26.png)
        - 迁移transformers==4.37.0版本内cache_utils,modeling_attn_mask_utils,modeling_outputs和utils中移动至modeling_qwen2_vacc.py

    - [configuration_qwen2_vacc.py](./build_in/source_code/configuration_qwen2_vacc.py)
        - 修改对于相关依赖的导入方式
        ![](../../images/llm/qwen/Snipaste_2024-04-15_17-31-20.png)
    - [quantization_vacc.py](./build_in/source_code/quantization_vacc.py)
        - Qwen2ForCausalLM添加quantize方法，支持per_channel int8量化
        ![](../../images/llm/qwen/Snipaste_2025-03-20_20-10-41.png)
    - [config_vacc.json](./build_in/source_code/config_vacc.json)
        - 添加_attn_implementation选项，并将其只配置为eager；并添加auto_map选项
        ![](../../images/llm/qwen/Snipaste_2024-04-15_17-34-02.png)
    - 将以上修改后文件，放置与原始权重目录下（注意不同子模型，对应修改config_vacc.json文件）

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
    - v1.5/v2/v2.5模型，编译配置一致
    - [hf_qwen1.5_fp16.yaml](./build_in/build/hf_qwen1.5_fp16.yaml)
    - [hf_qwen1.5_int8.yaml](./build_in/build/hf_qwen1.5_int8.yaml)

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`
    
    ```bash
    vamc compile ./build_in/build/hf_qwen1.5_fp16.yaml
    vamc compile ./build_in/build/hf_qwen1.5_int8.yaml
    ```


### step.4 模型推理
1. 参考大模型部署推理工具：[vastgenx: v1.1.0+](../../docs/vastgenx/README.md)

### Tips
- **LLM模型请先查看概要指引**，[Tips🔔](../README.md)
- 建议环境配置
    ```bash
    protobuf==3.20.3
    torch==2.1.0
    onnx==1.14.0
    onnxsim==0.4.28
    onnxruntime==1.13.1
    accelerate==0.25.0
    transformers==4.40
    ```

## Pytorch Deploy

### step.1 模型准备
|  models |    demo_code    | tips |
| :------ | :------: | :------: | 
|[Qwen/Qwen1.5-0.5B-Chat](https://hf-mirror.com/Qwen/Qwen1.5-0.5B-Chat) | [demo](./pytorch/demo/qwen1.5.py) | - |
|[Qwen/Qwen1.5-1.8B-Chat](https://hf-mirror.com/Qwen/Qwen1.5-1.8B-Chat) | [demo](./pytorch/demo/qwen1.5.py) | - |

### step.2 模型推理
- 基于`torch_vacc`在`VA16`硬件下推理，一般基于官方demo进行适当修改，参见上表`demo_code`部分
