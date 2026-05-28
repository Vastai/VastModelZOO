# GLM-OCR Quant

## FP8
- 量化工具

> 基于：[llm-compressor](https://github.com/vllm-project/llm-compressor.git)

```bash
conda create --name glm_ocr_quant python==3.10
# use main branch e688d87
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
# upgrade transformers, ignore error
pip install transformers==5.3.0 -i https://mirrors.ustc.edu.cn/pypi/web/simple
pip install torchvision==0.27.0 -i https://mirrors.ustc.edu.cn/pypi/web/simple
```

- 量化实现

```python
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="./weights/GLM-OCR",
        help="Path to the model",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./quant_weights/",
        help="Path to save the quantized model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path {args.model_path} does not exist.")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    tokenizer = AutoProcessor.from_pretrained(args.model_path)

    # Configure the quantization algorithm and scheme.
    # In this case, we:
    #   * quantize the weights to fp8 with per block 128x128 via ptq
    #   * quantize the input activations to fp8 with dynamic per token per group 128
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_BLOCK",
        ignore=[
            "lm_head",
            "re:.*input_layernorm$",
            "re:.*post_attention_layernorm$",
            "re:model.visual.*",
        ],
    )

    # Apply quantization.
    oneshot(model=model, recipe=recipe)

    # Save to disk
    save_name = args.model_path.split("/")[-1] + "-FP8"
    save_dir = os.path.join(args.save_path, save_name)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
```

- config.json 配置修改

> 将量化后的模型的 config.json 替换成如下


<details><summary><b>config.json</summary>

```json
{
  "architectures": [
    "GlmOcrForConditionalGeneration"
  ],
  "dtype": "bfloat16",
  "image_end_token_id": 59257,
  "image_start_token_id": 59256,
  "image_token_id": 59280,
  "model_type": "glm_ocr",
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "modules_to_not_convert": [
      "model.visual.blocks.0.attn.q_proj",
      "model.visual.blocks.0.attn.k_proj",
      "model.visual.blocks.0.attn.v_proj",
      "model.visual.blocks.0.attn.proj",
      "model.visual.blocks.0.mlp.gate_proj",
      "model.visual.blocks.0.mlp.up_proj",
      "model.visual.blocks.0.mlp.down_proj",
      "model.visual.blocks.1.attn.q_proj",
      "model.visual.blocks.1.attn.k_proj",
      "model.visual.blocks.1.attn.v_proj",
      "model.visual.blocks.1.attn.proj",
      "model.visual.blocks.1.mlp.gate_proj",
      "model.visual.blocks.1.mlp.up_proj",
      "model.visual.blocks.1.mlp.down_proj",
      "model.visual.blocks.2.attn.q_proj",
      "model.visual.blocks.2.attn.k_proj",
      "model.visual.blocks.2.attn.v_proj",
      "model.visual.blocks.2.attn.proj",
      "model.visual.blocks.2.mlp.gate_proj",
      "model.visual.blocks.2.mlp.up_proj",
      "model.visual.blocks.2.mlp.down_proj",
      "model.visual.blocks.3.attn.q_proj",
      "model.visual.blocks.3.attn.k_proj",
      "model.visual.blocks.3.attn.v_proj",
      "model.visual.blocks.3.attn.proj",
      "model.visual.blocks.3.mlp.gate_proj",
      "model.visual.blocks.3.mlp.up_proj",
      "model.visual.blocks.3.mlp.down_proj",
      "model.visual.blocks.4.attn.q_proj",
      "model.visual.blocks.4.attn.k_proj",
      "model.visual.blocks.4.attn.v_proj",
      "model.visual.blocks.4.attn.proj",
      "model.visual.blocks.4.mlp.gate_proj",
      "model.visual.blocks.4.mlp.up_proj",
      "model.visual.blocks.4.mlp.down_proj",
      "model.visual.blocks.5.attn.q_proj",
      "model.visual.blocks.5.attn.k_proj",
      "model.visual.blocks.5.attn.v_proj",
      "model.visual.blocks.5.attn.proj",
      "model.visual.blocks.5.mlp.gate_proj",
      "model.visual.blocks.5.mlp.up_proj",
      "model.visual.blocks.5.mlp.down_proj",
      "model.visual.blocks.6.attn.q_proj",
      "model.visual.blocks.6.attn.k_proj",
      "model.visual.blocks.6.attn.v_proj",
      "model.visual.blocks.6.attn.proj",
      "model.visual.blocks.6.mlp.gate_proj",
      "model.visual.blocks.6.mlp.up_proj",
      "model.visual.blocks.6.mlp.down_proj",
      "model.visual.blocks.7.attn.q_proj",
      "model.visual.blocks.7.attn.k_proj",
      "model.visual.blocks.7.attn.v_proj",
      "model.visual.blocks.7.attn.proj",
      "model.visual.blocks.7.mlp.gate_proj",
      "model.visual.blocks.7.mlp.up_proj",
      "model.visual.blocks.7.mlp.down_proj",
      "model.visual.blocks.8.attn.q_proj",
      "model.visual.blocks.8.attn.k_proj",
      "model.visual.blocks.8.attn.v_proj",
      "model.visual.blocks.8.attn.proj",
      "model.visual.blocks.8.mlp.gate_proj",
      "model.visual.blocks.8.mlp.up_proj",
      "model.visual.blocks.8.mlp.down_proj",
      "model.visual.blocks.9.attn.q_proj",
      "model.visual.blocks.9.attn.k_proj",
      "model.visual.blocks.9.attn.v_proj",
      "model.visual.blocks.9.attn.proj",
      "model.visual.blocks.9.mlp.gate_proj",
      "model.visual.blocks.9.mlp.up_proj",
      "model.visual.blocks.9.mlp.down_proj",
      "model.visual.blocks.10.attn.q_proj",
      "model.visual.blocks.10.attn.k_proj",
      "model.visual.blocks.10.attn.v_proj",
      "model.visual.blocks.10.attn.proj",
      "model.visual.blocks.10.mlp.gate_proj",
      "model.visual.blocks.10.mlp.up_proj",
      "model.visual.blocks.10.mlp.down_proj",
      "model.visual.blocks.11.attn.q_proj",
      "model.visual.blocks.11.attn.k_proj",
      "model.visual.blocks.11.attn.v_proj",
      "model.visual.blocks.11.attn.proj",
      "model.visual.blocks.11.mlp.gate_proj",
      "model.visual.blocks.11.mlp.up_proj",
      "model.visual.blocks.11.mlp.down_proj",
      "model.visual.blocks.12.attn.q_proj",
      "model.visual.blocks.12.attn.k_proj",
      "model.visual.blocks.12.attn.v_proj",
      "model.visual.blocks.12.attn.proj",
      "model.visual.blocks.12.mlp.gate_proj",
      "model.visual.blocks.12.mlp.up_proj",
      "model.visual.blocks.12.mlp.down_proj",
      "model.visual.blocks.13.attn.q_proj",
      "model.visual.blocks.13.attn.k_proj",
      "model.visual.blocks.13.attn.v_proj",
      "model.visual.blocks.13.attn.proj",
      "model.visual.blocks.13.mlp.gate_proj",
      "model.visual.blocks.13.mlp.up_proj",
      "model.visual.blocks.13.mlp.down_proj",
      "model.visual.blocks.14.attn.q_proj",
      "model.visual.blocks.14.attn.k_proj",
      "model.visual.blocks.14.attn.v_proj",
      "model.visual.blocks.14.attn.proj",
      "model.visual.blocks.14.mlp.gate_proj",
      "model.visual.blocks.14.mlp.up_proj",
      "model.visual.blocks.14.mlp.down_proj",
      "model.visual.blocks.15.attn.q_proj",
      "model.visual.blocks.15.attn.k_proj",
      "model.visual.blocks.15.attn.v_proj",
      "model.visual.blocks.15.attn.proj",
      "model.visual.blocks.15.mlp.gate_proj",
      "model.visual.blocks.15.mlp.up_proj",
      "model.visual.blocks.15.mlp.down_proj",
      "model.visual.blocks.16.attn.q_proj",
      "model.visual.blocks.16.attn.k_proj",
      "model.visual.blocks.16.attn.v_proj",
      "model.visual.blocks.16.attn.proj",
      "model.visual.blocks.16.mlp.gate_proj",
      "model.visual.blocks.16.mlp.up_proj",
      "model.visual.blocks.16.mlp.down_proj",
      "model.visual.blocks.17.attn.q_proj",
      "model.visual.blocks.17.attn.k_proj",
      "model.visual.blocks.17.attn.v_proj",
      "model.visual.blocks.17.attn.proj",
      "model.visual.blocks.17.mlp.gate_proj",
      "model.visual.blocks.17.mlp.up_proj",
      "model.visual.blocks.17.mlp.down_proj",
      "model.visual.blocks.18.attn.q_proj",
      "model.visual.blocks.18.attn.k_proj",
      "model.visual.blocks.18.attn.v_proj",
      "model.visual.blocks.18.attn.proj",
      "model.visual.blocks.18.mlp.gate_proj",
      "model.visual.blocks.18.mlp.up_proj",
      "model.visual.blocks.18.mlp.down_proj",
      "model.visual.blocks.19.attn.q_proj",
      "model.visual.blocks.19.attn.k_proj",
      "model.visual.blocks.19.attn.v_proj",
      "model.visual.blocks.19.attn.proj",
      "model.visual.blocks.19.mlp.gate_proj",
      "model.visual.blocks.19.mlp.up_proj",
      "model.visual.blocks.19.mlp.down_proj",
      "model.visual.blocks.20.attn.q_proj",
      "model.visual.blocks.20.attn.k_proj",
      "model.visual.blocks.20.attn.v_proj",
      "model.visual.blocks.20.attn.proj",
      "model.visual.blocks.20.mlp.gate_proj",
      "model.visual.blocks.20.mlp.up_proj",
      "model.visual.blocks.20.mlp.down_proj",
      "model.visual.blocks.21.attn.q_proj",
      "model.visual.blocks.21.attn.k_proj",
      "model.visual.blocks.21.attn.v_proj",
      "model.visual.blocks.21.attn.proj",
      "model.visual.blocks.21.mlp.gate_proj",
      "model.visual.blocks.21.mlp.up_proj",
      "model.visual.blocks.21.mlp.down_proj",
      "model.visual.blocks.22.attn.q_proj",
      "model.visual.blocks.22.attn.k_proj",
      "model.visual.blocks.22.attn.v_proj",
      "model.visual.blocks.22.attn.proj",
      "model.visual.blocks.22.mlp.gate_proj",
      "model.visual.blocks.22.mlp.up_proj",
      "model.visual.blocks.22.mlp.down_proj",
      "model.visual.blocks.23.attn.q_proj",
      "model.visual.blocks.23.attn.k_proj",
      "model.visual.blocks.23.attn.v_proj",
      "model.visual.blocks.23.attn.proj",
      "model.visual.blocks.23.mlp.gate_proj",
      "model.visual.blocks.23.mlp.up_proj",
      "model.visual.blocks.23.mlp.down_proj",
      "model.visual.merger.proj",
      "model.visual.merger.gate_proj",
      "model.visual.merger.up_proj",
      "model.visual.merger.down_proj",
      "lm_head"
    ],
    "quant_method": "fp8",
    "weight_block_size": [
      128,
      128
    ]
  },
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "dtype": "bfloat16",
    "eos_token_id": [
      59246,
      59253
    ],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1536,
    "initializer_range": 0.02,
    "intermediate_size": 4608,
    "max_position_embeddings": 131072,
    "model_type": "glm_ocr_text",
    "num_attention_heads": 16,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "num_nextn_predict_layers": 1,
    "pad_token_id": 59246,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {
      "mrope_section": [
        16,
        24,
        24
      ],
      "partial_rotary_factor": 1.0,
      "rope_theta": 10000,
      "rope_type": "default"
    },
    "tie_word_embeddings": false,
    "use_cache": true,
    "vocab_size": 59392
  },
  "tie_word_embeddings": false,
  "transformers_version": "5.3.0",
  "video_end_token_id": 59259,
  "video_start_token_id": 59258,
  "video_token_id": 59281,
  "vision_config": {
    "attention_bias": true,
    "attention_dropout": 0.0,
    "depth": 24,
    "dtype": "bfloat16",
    "hidden_act": "silu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 1024,
    "image_size": 336,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "model_type": "glm_ocr_vision",
    "num_heads": 16,
    "out_hidden_size": 1536,
    "patch_size": 14,
    "rms_norm_eps": 1e-05,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2
  }
}
```
</details>