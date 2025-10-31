# Convert Pretrain Model

运行 pt2torchscript.py
```bash
cd VastModelZOO/nlp/named_entity_recognition/xlmroberta/huggingface/source_code/pretrain_model

python pt2torchscript.py \
    --model_name_or_path ./xlmroberta_base_zh_ner-256  \
    --save_path ./xlmroberta_base_zh_ner-256.torchscript.pt \
    --seq_length 256
```
- model_name_or_path:  [step.1](../finetune/huggingface_xlmroberta_ner.md) 或者 huggingface 微调的预训练模型，可以在线和离线加载
- seq_length: 最大序列长度
- save_path：torchscript 权重输出路径
