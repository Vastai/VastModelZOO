# Huggingface RoBerta-ner Finetune 


运行 roberta_ner_finetune.py
```bash
cd VastModelZOO/nlp/named_entity_recognition/roberta/huggingface/source_code/finetune

python roberta_ner_finetune.py \
    --model_name_or_path path/to/pretrain_model/chinese-roberta-wwm-ext \
    --output_dir path/to/output/roberta_wwm_ext_ner_people_daily \
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