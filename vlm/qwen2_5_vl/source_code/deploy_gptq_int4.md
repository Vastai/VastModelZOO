## TVM_VACC部署

- qwen-vl int4系列模型部署采用混合部署方式，即Visual模块采用FP16精度TVM_VACC方式部署，LLM模块采用int4精度直接抽取的方式使用vastgenx部署。

### step.1 模型准备

1. 下载模型权重

    | models  | tips |
    | :---: | :--: |
    | [Qwen/Qwen2.5-VL-7B-Instruct-GPTQ-Int4](https://modelscope.cn/models/ChineseAlpacaGroup/Qwen2.5-VL-7B-Instruct-GPTQ-Int4) | [modeling_qwen2_5_vl_vacc.py](./modify_modeling/modeling_qwen2_5_vl_vacc.py) |

2. 网络修改
    - 为了方便部署`Qwen2.5-VL`系列模型，在官方源码的基础上，需要对`modeling_qwen2_5_vl.py`进行适当修改
    - 修改后：[modeling_qwen2_5_vl_vacc.py](./modify_modeling/modeling_qwen2_5_vl_vacc.py)

3. Config修改
    - Visual部分延续TVM_VACC使用FP16精度部署，因此需对原始权重的`config.json`进行适当修改，以便vamc正常编译
    - 修改后：[config_vacc.json](./modify_modeling/config_int4_vacc.json)

### step.2 获取数据集

- vlm模型基于`evalscope`工具进行精度测评，数据集参考：[supported_dataset](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset.html#vlmevalkit)


### step.3 模型转换抽取

1. visual模块根据具体模型修改配置文件：
    - [qwen2_5_vl_visual_gptq.yaml](../build_in/build/qwen2_5_vl_visual_gptq.yaml)

    ```bash
    cd qwen2_5_vl
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/qwen2_5_vl_visual_gptq.yaml
    ```
2. llm模块抽取：
    - 修改相应权重路径并执行 [contract_llm.py](../source_code/contract_llm/contract_llm.py)

    ```bash
    python3 ../source_code/contract_llm/
    ```

### step.4 模型推理
1. 获取大模型部署推理工具：[vastgenx](../../../docs/vastgenx/README.md)
2. 精度评估参考：[vastgenx-精度测试](../../../docs/vastgenx/README.md#-精度测试)
- 本例参考：

```bash
vastgenx serve --model Qwen2.5-VL-7B-Instruct-GPTQ-Int4-llm \
--vit_model vamc_results/qwen2_5_vl_7b_visual_gptq_fp16 \
--port 9900 \
--llm_devices "[0,1]" \
--tensor_parallel_size 2
```
3. 性能评估参考：[vastgenx-性能测试](../../../docs/vastgenx/README.md#-性能测试)


### Tips
- 混合部署中，拆分为两个模型进行推理，Visual部分不切分，LLM部分直接使用Vastgenx进行TP切分
- 仅验证了Qwen2.5-VL-7B-Instruct-GPTQ-Int4，transformers版本需要较新
- 精度测试可测试多个数据集：
    - MMBench_DEV_EN
    - MMBench_DEV_CN
    - MMMU_DEV_VAL
    - OCRBench
- Qwen2.5_VL依赖的transformers版本较新
    ```
    transformers==4.51.3
    huggingface-hub==0.30.2
    ```

- input_ids转为embedding部分，需要使用从预训练模型获取的embedding权重：`embed_tokens.bin`
    ```python
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

    model_name = "./vlm/Qwen/Qwen2.5-VL-7B-Instruct-GPTQ-Int4"

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    embed_tokens = model.embed_tokens.weight.data
    embed_tokens.cpu().numpy().tofile("embed_tokens.bin")
    print(f"embed_tokens shape: {embed_tokens.shape}")
    ```

- 抽取llm模块得到的Qwen2.5-VL-7B-Instruct-GPTQ-Int4-llm 所对应的 `config.json`，需要进行适当修改以满足llm模块，修改后的：[config_int4_llm.json](./modify_modeling/config_int4_llm.json)