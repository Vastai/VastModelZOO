# Convert Pretrain Model

运行 pt2torchscript.py
```bash
cd path/to/modelzoo/sentence_classification/bert/huggingface/source_code/pretrain_model

python pt2torchscript.py \
    --model_name_or_path path/to/pretrain_model \
    --save_path path/to/output/bert_cls/bert_base_cls.torchscript.pt \
    --seq_length 128 \
```
- model_name_or_path: [step.1](../finetune/huggingface_bert_mrpc.md) 或者 huggingface 微调的预训练模型，可以在线和离线加载
- seq_length: 最大序列长度
- save_path：torchscript 权重输出路径

