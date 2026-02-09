# Qwen3-Dense AutoRound-GPTQ

> - 使用工具：[intel/auto-round](https://github.com/intel/auto-round)
> - 参考官方量化流程：[Intel/Qwen3-14B-int4-AutoRound-gptq-inc](https://www.modelscope.cn/models/Intel/Qwen3-14B-int4-AutoRound-gptq-inc/summary)

```json
"quantization_config": {
    "act_bits": 16,
    "act_data_type": "int",
    "act_dynamic": true,
    "act_group_size": 128,
    "act_sym": true,
    "amp": true,
    "autoround_version": "0.5.1",
    "batch_size": 8,
    "bits": 4,
    "damp_percent": 0.01,
    "data_type": "int",
    "desc_act": false,
    "enable_minmax_tuning": true,
    "enable_norm_bias_tuning": false,
    "enable_quanted_input": true,
    "gradient_accumulate_steps": 1,
    "group_size": 128,
    "iters": 1000,
    "low_gpu_mem_usage": true,
    "lr": 0.001,
    "minmax_lr": 0.001,
    "nsamples": 512,
    "quant_method": "gptq",
    "scale_dtype": "torch.float16",
    "seqlen": 2048,
    "super_bits": null,
    "super_group_size": null,
    "sym": true,
    "to_quant_block_names": null,
    "true_sequential": false
}
"torch_dtype": "float16",
```

## Install
- 参考依赖：[requirements.txt](./requirements.txt)

```bash
# auto_gptq==0.7.1 auto_round==0.5.1
pip install -r requirements.txt
```

## Guide

1. Quant

    ```bash
    # 量化到 Int4
    auto-round-best \                                                             
    --model weights/Qwen3-14B \
    --device 0 \
    --group_size 128 \
    --bits 4 \
    --format 'auto_gptq' \
    --output_dir "./Qwen3-14B-Int4"


    # 量化到 Int8
    auto-round-best \                                                             
    --model weights/Qwen3-14B \
    --device 0 \
    --group_size 128 \
    --bits 8 \
    --format 'auto_gptq' \
    --output_dir "./Qwen3-14B-Int8"
    ```

2. Eval

- 使用evalscope测试精度信息：[precision_llm.py](../../../../tools/evalscope/precision_llm.py)