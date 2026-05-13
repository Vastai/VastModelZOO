# Qwen3-30B-A3B GPTQ

> - 分析 [Qwen3-30B-A3B-GPTQ-Int4](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B-GPTQ-Int4) 可知量化参数 -> GPTQ-W4A16, 不量化 `mlp.gate` 和 `lm_head`
> - 使用`gptqmodel`库进行量化

```bash
Qwen3MoeForCausalLM(
  (model): Qwen3MoeModel(
    (embed_tokens): Embedding(151936, 2048)
    (layers): ModuleList(
      (0-47): 48 x Qwen3MoeDecoderLayer(
        (self_attn): Qwen3MoeAttention(
          (q_proj): Linear(in_features=2048, out_features=4096, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=4096, out_features=2048, bias=False)
          (q_norm): Qwen3MoeRMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3MoeRMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MoeSparseMoeBlock(
          (gate): Linear(in_features=2048, out_features=128, bias=False)
          (experts): ModuleList(
            (0-127): 128 x Qwen3MoeMLP(
              (gate_proj): Linear(in_features=2048, out_features=768, bias=False)
              (up_proj): Linear(in_features=2048, out_features=768, bias=False)
              (down_proj): Linear(in_features=768, out_features=2048, bias=False)
              (act_fn): SiLUActivation()
            )
          )
        )
        (input_layernorm): Qwen3MoeRMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen3MoeRMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen3MoeRMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen3MoeRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)
```

```json
"quantization_config": {
        "bits": 4,
        "checkpoint_format": "gptq",
        "damp_percent": 0.01,
        "desc_act": false,
        "group_size": 128,
        "model_file_base_name": null,
        "model_name_or_path": null,
        "quant_method": "gptq",
        "static_groups": false,
        "sym": true,
        "true_sequential": true
    }
```

## Install
- gptqmodel==4.0.0
- vllm==0.11.1
- eavlscope==0.17.1
- transformers==4.57.3

## Guide
> 原始模型：Qwen3-30B-A3B-Instruct-2507/Qwen3-30B-A3B-Thinking-2507

1. Eval BF16

    ```bash
    CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --max-model-len 131072 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8012 \
    --served-model-name Qwen3-30B-A3B-Thinking-2507 \
    --enforce-eager \
    --reasoning-parser deepseek_r1

    # instruct
    python eval.py --temperature 0.7 --top_p 0.8  --max_tokens 16384 --model Qwen3-30B-A3B-Instruct-2507

    # thinking
    python eval.py --temperature 0.6 --top_p 0.95  --max_tokens 81920 --model Qwen3-30B-A3B-Thinking-2507
    ```

2. Quant

    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python main.py
    ```

3. Eval Int4

    ```bash
    CUDA_VISIBLE_DEVICES=0,1 vllm serve ./weights/Qwen3-30B-A3B-Thinking-2507-GPTQ-Int4 \
    --max-model-len 131072 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8012 \
    --served-model-name Qwen3-30B-A3B-Thinking-2507-GPTQ-Int4 \
    --enforce-eager \
    --reasoning-parser deepseek_r1

    # instruct
    python eval.py --temperature 0.7 --top_p 0.8  --max_tokens 16384 --model Qwen3-30B-A3B-Instruct-2507--GPTQ-Int4

    # thinking
    python eval.py --temperature 0.6 --top_p 0.95  --max_tokens 81920 --model Qwen3-30B-A3B-Thinking-2507-GPTQ-Int4
    ```

## Accuracy

- Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4

| **id** | **Dataset**     | **Metric**              | **Samples** | **nv-score-bf16** | **nv-score-fp8** | **nv-score-w4a16** | **nv-score-w4a16-no-gate-linear** |
| ------ | --------------- | ----------------------- | ----------- | ----------------- | ---------------- | ------------------ | --------------------------------- |
| **1**  | aime25          | AveragePass@1           | 30          | 0.5666            | 0.594417         | **0.5**            | 0.6333                            |
| **2**  | gpqa_diamond    | AveragePass@1           | 198         | 0.667             | 0.64848          | **0.6263**         | **0.6364**                        |
| **3**  | mmlu_pro        | AverageAccuracy         | 1196        | 0.7751            | 0.7755           | **0.7466**         | 0.7676                            |
| **4**  | ifeval          | prompt_level_strict_acc | 541         | 0.8355            | 0.83952          | 0.8429             | 0.8336                            |
| **5**  | live_code_bench | Pass@1                  | 1055        | 0.563             | 0.56358          | **0.491**          | **0.5213**                        |

- Qwen3-30B-A3B-Thinking-2507-GPTQ-Int4

| **id** | **Dataset**     | **Metric**              | **Samples** | **nv-score-bf16** | **nv-score-w4a16** | **nv-score-w4a16-no-gate-linear** |
| ------ | --------------- | ----------------------- | ----------- | ----------------- | ------------------ | --------------------------------- |
| **1**  | aime25          | AveragePass@1           | 30          | 0.7666            | 0.8334             | 0.9334                            |
| **2**  | gpqa_diamond    | AveragePass@1           | 198         | 0.6869            | **0.6313**         | 0.7071                            |
| **3**  | mmlu_pro        | AverageAccuracy         | 1196        | 0.7784            | 0.7617             | 0.7734                            |
| **4**  | ifeval          | prompt_level_strict_acc | 541         | 0.8521            | 0.8503             | 0.8521                            |
| **5**  | live_code_bench | Pass@1                  | 1055        | 0.7922            | **0.7687**         | **0.7782**                        |