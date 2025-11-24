## build_in部署

### step.1 模型准备

1. 下载模型权重

    | models  | tips |
    | :---: | :--: |
    | [reducto/RolmOCR](https://hf-mirror.com/reducto/RolmOCR) |[modeling_qwen2_5_vl_vacc.py](./modify_modeling/modeling_qwen2_5_vl_vacc.py) |
    

2. 网络修改
    - 为了方便部署`Qwen2.5-VL`系列模型，在官方源码的基础上，需要对`modeling_qwen2_5_vl.py`进行适当修改
    - 修改后：[modeling_qwen2_5_vl_vacc.py](./modify_modeling/modeling_qwen2_5_vl_vacc.py)


### step.2 获取数据集

- vlm模型基于`evalscope`工具进行精度测评，数据集参考：[supported_dataset](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset.html#vlmevalkit)

### step.3 模型转换
1. 根据具体模型修改配置文件：
    - [rolmocr_llm.yaml](../build_in/build/rolmocr_llm.yaml)
    - [rolmocr_visual.yaml](../build_in/build/rolmocr_visual.yaml)

    ```bash
    cd rolmocr
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/rolmocr_llm.yaml
    vamc compile ../build_in/build/rolmocr_visual.yaml
    ```

### step.4 模型推理
1. 获取大模型部署推理工具：[vastgenx](../../../docs/vastgenx/README.md)
2. 精度评估参考：[vastgenx-精度测试](../../../docs/vastgenx/README.md#-精度测试)
- 本例参考：

```bash
vastgenx serve --model vamc_results/rolmocr_llm_fp16 \
--vit_model vamc_results/rolmocr_visual_fp16 \
--port 9900 \
--llm_devices "[4,5,6,7]" \
--vit_devices "[3]" \
--min_pixels 78400 \
--max_pixels 921600
```
```
[VACC]:
{
    "Text Recognition": 263,
    "Scene Text-centric VQA": 176,
    "Doc-oriented VQA": 170,
    "Key Information Extraction": 178,
    "Handwritten Mathematical Expression Recognition": 62,
    "Final Score": 849,
    "Final Score Norm": 84.9
}
```

```
[NV]:
{
    "Text Recognition": 260,
    "Scene Text-centric VQA": 176,
    "Doc-oriented VQA": 169,
    "Key Information Extraction": 186,
    "Handwritten Mathematical Expression Recognition": 66,
    "Final Score": 857,
    "Final Score Norm": 85.7
}
```

3. 性能评估参考：[vastgenx-性能测试](../../../docs/vastgenx/README.md#-性能测试)



### Tips
- VACC部署中，拆分为两个模型进行推理，Visual部分不切分，LLM部分可TP切分
- Visual部分
    - 计算最佳图像尺寸Smart_Resize操作在CPU上实现，其它缩放、均值方差、归一化在VDSP上实现
    - visual部分的旋转位置编码在VDSP自定义算子上实现，需要把sin/cos置前，通过参数传入
    - 模型输出`image_embeds`数据量太大([1280,3584])，由于硬件限制，compile会切成6份输出，前5份[1280,640]，最后一份[1280,384]
    - 模型编译耗时7小时，推理时加载需6分钟（和attention_split_num及ffn_split_num参数相关）
    - 关于visual部分尺寸限制，以720P(720x1280)为例
        - 首先，宽高将缩放至最接近28倍数值上（728x1288）；
        - 在VIT计算时将转换为(728//14) x (1288//14) = 4784 patch
        - 在编译visual部分三件套时，设置稍大于4784的值（且需为16的倍数），如5120
        - 在visual部分转换为token时，有相邻patch-merge操作(2x2)，最终visual-token=5120//4=1280
        - 因此，在llm部分的prefill阶段seq-length需大于1280，如设置为2048
- LLM部分
    - 3D旋转位置编码MROPE的`get_rope_index`在CPU上实现
- Visual部分基于VSX推理，bs=1，每次只能处理一张图片；通过消息队列向基于vast_llm推理的LLM部分传递，最后输出文本回答
- 精度测试可测试数据集：
    - OCRBench
- RolmOCR依赖的transformers版本较新：
    ```
    # 注意，transformers官方在python3.8下只支持到4.47；所有此模型需要python3.10
    transformers==4.51.3
    huggingface-hub==0.30.2
    ```

- input_ids转为embedding部分，需要使用从预训练模型获取的embedding权重：`emb_tokens.bin`
    ```python
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

    model_name = "./vlm/reducto/RolmOCR"

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



