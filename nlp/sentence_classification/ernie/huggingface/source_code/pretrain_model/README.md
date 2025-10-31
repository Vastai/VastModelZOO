# Convert Pretrain Model

运行 pt2torchscript.py
```bash
cd VastModelZOO/nlp/sentence_classification/ernie/huggingface/source_code/pretrain_model

python pt2torchscript.py \
    --model_name_or_path path/to/pretrain_model/ernie2.0-base-en \
    --save_path path/to/output/bert_cls/ernie2_base_en_cls.torchscript.pt \
    --seq_length 128 \
```
- model_name_or_path: [step.1](../finetune/huggingface_ernie_people_daily.md) 或者 huggingface 微调的预训练模型，可以在线和离线加载
- seq_length: 最大序列长度
- save_path：torchscript 权重输出路径

