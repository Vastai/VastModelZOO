# Huggingface Bert-ner Finetune 

## finetune China People daily 中文实体命名识别数据集
运行 ernie_ner_finetune.py
```bash
cd VastModelZOO/nlp/named_entity_recognition/ernie/huggingface/source_code/finetune

python bert_ner_finetune.py \
    --model_name_or_path path/to/pretrain_model/ernie2.0-base-en \
    --output_dir path/to/output/ernie2_base_en_ner_people_daily \
    --num_labels 7 \
    --seq_length 256 \
    --do_train \
    --do_eval \
    --data_dir path/to/datasets/china-people-daily-ner-corpus
```
- model_name_or_path: huggingface 网站上的预训练模型，可以在线和离线加载
- output_dir：模型权重输出路径
- num_labels: 输出的类别
- seq_length： 最大输入序列长度
- do_train： 是否训练
- do_eval： 是否进行最终评估，如果是，则将评估结果保存在 `output_dir` 路径下
- data_dir： 中国人民日报数据


</br>

### **Tips**

如果是下载的 huggingface 网站上的 ERNIE 3.0 系列的模型进行微调时，需要在运行 `ernie_ner_finetune.py` 之前修改预训练模型里的 `config.json`。
```bash
cd path/to/pretrain_model/ernie3.0-base-zh

vim config.json

{
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "max_position_embeddings": 2048,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "task_type_vocab_size": 3,
    "type_vocab_size": 4,
    "use_task_id": false,   # 修改为 false
    "vocab_size": 40000,
    "layer_norm_eps": 1e-05,
    "model_type": "ernie",
    "architectures": [
        "ErnieForMaskedLM"
    ],
    "intermediate_size": 3072
}
```

打开 `config.json` ，修改 `"use_task_id"`:false, 保存