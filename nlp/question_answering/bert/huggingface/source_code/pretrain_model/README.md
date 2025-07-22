# Convert Pretrain Model

## step.1 修改 transformers 代码

- 进入本地 transformers 库环境，修改 bert 模型输出代码

    ```bash
    cd /path/to/anaconda3/envs/bert_mult/lib/python3.7/site-packages/transformers/src/transformers/models/bert/

    vim modeling_bert.py
    ```
- 修改 1865 - 1872行 代码为：
    ```python
    logits = self.qa_outputs(sequence_output)

    if self.training:
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
    else:
        return logits
    ```

## step.2 输出 torchscript 模型
运行 pt2torchscript.py

```bash
cd path/to/algorithm_modelzoo/question_answering/bert/huggingface/source_code/pretrain_model

python pt2torchscript.py  \
    --model_name_or_path ./bert_base_en_qa-384 \
    --seq_length 384 \
    --save_path ./bert_base_en_qa-384.torchscript.pt
```
- model_name_or_path: [step.1](../finetune/huggingface_bert_mrpc.md) 或者 huggingface 微调的预训练模型，可以在线和离线加载
- seq_length: 最大序列长度
- save_path：torchscript 权重输出路径