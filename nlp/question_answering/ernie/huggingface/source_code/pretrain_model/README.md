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
cd path/to/VastModelZOO/nlp/question_answering/ernie/huggingface/source_code/pretrain_model

python pt2torchscript.py  \ 
    --model_name_or_path /path/to/runs/SQuAD_for_qa_bert_base_uncased_384/checkpoint-3500  \
    --seq_length 384\
    --save_path /path/to/runs/SQuAD_for_qa_bert_base_uncased_384/checkpoint-3500/bert_base_en_qa-384.torchscript.pt


python pt2torchscript.py  \
    --model_name_or_path pretrain/ernie1.0_base_zh_qa-384/  \
    --seq_length 384 \
    --save_path ernie1.0_base_zh_qa-384.torchscript.pt


python pt2torchscript.py  \
    --model_name_or_path pretrain/ernie2.0_base_en_qa-384/  \
    --seq_length 384 \
    --save_path ernie2.0_base_en_qa-384.torchscript.pt

python pt2torchscript.py  \
    --model_name_or_path pretrain/ernie2.0_large_en_qa-384/  \
    --seq_length 384 \
    --save_path ernie2.0_large_en_qa-384.torchscript.pt

python pt2torchscript.py  \
    --model_name_or_path pretrain/ernie3.0_base_zh_qa-384/  \
    --seq_length 384 \
    --save_path ernie3.0_base_zh_qa-384.torchscript.pt

python pt2torchscript.py  \
    --model_name_or_path pretrain/ernie3.0_medium_zh_qa-384/  \
    --seq_length 384 \
    --save_path ernie3.0_medium_zh_qa-384.torchscript.pt

python pt2torchscript.py  \
    --model_name_or_path pretrain/ernie3.0_xbase_zh_qa-384/  \
    --seq_length 384 \
    --save_path ernie3.0_xbase_zh_qa-384.torchscript.pt


```
