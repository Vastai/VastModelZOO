# Huggingface Bert-ner Finetune 


运行 bert_ner_finetune.py
```bash
cd VastModelZOO/nlp/named_entity_recognition/bert/huggingface/source_code/finetune

python bert_ner_finetune.py \
    --model_name_or_path path/to/pretrain_model/bert-base-chinese \
    --output_dir path/to/output/bert_base_chinese_ner_people_daily \
    --num_labels 7 \
    --seq_length 256 \
    --do_train \
    --do_eval \
    --data_dir path/to/datasets/china-people-daily-ner-corpus


python bert_ner_finetune.py \
    --model_name_or_path ~/weight/bert-base-chinese \
    --output_dir ./output/bert_base_chinese_ner_people_daily \
    --num_labels 7 \
    --seq_length 256 \
    --do_train \
    --do_eval \
    --data_dir ~/dataset/china-people-daily-ner-corpus/
```
- model_name_or_path: huggingface 网站上的预训练模型，可以在线和离线加载
- output_dir：模型权重输出路径
- num_labels: 输出的类别
- seq_length： 最大输入序列长度
- do_train： 是否训练
- do_eval： 是否进行最终评估，如果是，则将评估结果保存在 `output_dir` 路径下
- data_dir： 中国人民日报数据