# Convert Pretrain Model

## step.1 修改 transformers 代码
- transformers==4.26.1
- 修改forward返回值：[src/transformers/models/bert/modeling_bert.py](https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/bert/modeling_bert.py#L1865-L1868)

    ```python
    # logits = self.qa_outputs(sequence_output)
    # start_logits, end_logits = logits.split(1, dim=-1)
    # start_logits = start_logits.squeeze(-1).contiguous()
    # end_logits = end_logits.squeeze(-1).contiguous()

    # modify
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
cd VastModelZOO/nlp/question_answering/bert/huggingface/source_code/pretrain_model

python pt2torchscript.py  \
    --model_name_or_path ./bert_base_en_qa-384 \
    --seq_length 384 \
    --save_path ./bert_base_en_qa-384.torchscript.pt
```
- model_name_or_path: [step.1](../finetune/huggingface_bert_mrpc.md) 或者 huggingface 微调的预训练模型，可以在线和离线加载
- seq_length: 最大序列长度
- save_path：torchscript 权重输出路径