# Convert Pretrain Model

软件版本要求
```
pip install torch==1.6 transformers==4.30
```



运行 pt2torchscript.py
```bash
cd VastModelZOO/nlp/named_entity_recognition/bert/huggingface/source_code/pretrain_model


python pt2torchscript.py \
    --model_name_or_path /path/to/pretrain_model/bert_base_zh_ner-256 \
    --save_path ./bert_base_zh_ner-256.torchscript.pt \
    --seq_length 256 
```
- model_name_or_path:  [step.1](../finetune/huggingface_bert_peple_daily.md) 或者 huggingface 微调的预训练模型，可以在线和离线加载
- seq_length: 最大序列长度
- save_path：torchscript 权重输出路径
