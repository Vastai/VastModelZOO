# Convert Pretrain Model

运行 pt2torchscript.py
```bash
cd VastModelZOO/nlp/named_entity_recognition/roberta/huggingface/source_code/pretrain_model

python pt2torchscript.py \
    --model_name_or_path path/to/roberta_wwm_ext_ner_people_daily \
    --save_path path/to/output/roberta_wwm_ext_ner_people_daily/roberta_wwm_ext_ner_256.torchscript.pt \
    --seq_length 256 \


python pt2torchscript.py \
    --model_name_or_path ./roberta_wwm_ext_base_zh-256 \
    --save_path ./roberta_wwm_ext_ner_256.torchscript.pt \
    --seq_length 256 
```
- model_name_or_path:  [step.1](../finetune/huggingface_roberta_wwm_ext_peple_daily.md) 或者 huggingface 微调的预训练模型，可以在线和离线加载
- seq_length: 最大序列长度
- save_path：torchscript 权重输出路径

注意：pretrain_model 中 roberta_wwm_ext_base_zh-256 模型没有上传完整，用它来转模型会报错
```bash
ource_code/pretrain_model$ python pt2torchscript.py     --model_name_or_path ./roberta_wwm_ext_base_zh-256     --save_path ./roberta_wwm_ext_ner_256.torchscript.pt     --seq_length 256 
You are using a model of type bert to instantiate a model of type roberta. This is not supported for all configurations of models and can yield errors.
Traceback (most recent call last):
  File "pt2torchscript.py", line 109, in <module>
    model = RobertaForTokenClassification.from_pretrained(model_name_or_path, return_dict=False)
  File "/home/zhchen/miniconda3/envs/convert_torchscript/lib/python3.8/site-packages/transformers/modeling_utils.py", line 2881, in from_pretrained
    ) = cls._load_pretrained_model(
  File "/home/zhchen/miniconda3/envs/convert_torchscript/lib/python3.8/site-packages/transformers/modeling_utils.py", line 3113, in _load_pretrained_model
    raise ValueError(
ValueError: The state dictionary of the model you are trying to load is corrupted. Are you sure it was properly saved?
```