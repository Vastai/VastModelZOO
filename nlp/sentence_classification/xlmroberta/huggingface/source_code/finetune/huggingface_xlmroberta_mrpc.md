# HuggingFace RoBETRa-cls Finetune

运行 xlmroberta_cls_finetune.py
```bash
cd VastModelZOO/nlp/sentence_classification/xlmroberta/huggingface/source_code/finetune

python xlmroberta_cls_finetune.py \
    --model_name_or_path path/to/pretrain_model \
    --output_dir path/to/output/roberta_cls \
    --seq_length 128 \
    --do_train \
    --do_eval \
    --do_predict
```
- model_name_or_path: huggingface 网站上的预训练模型，可以在线和离线加载
- output_dir：模型权重输出路径
- seq_length： 最大输入序列长度
- do_train： 是否训练
- do_eval： 是否进行最终评估，如果是，则将评估结果保存在 `output_dir` 路径下
- do_predict： 是否进行预测， 如果是，则将预测结果和对应标签保存在 `output_dir` 路径下

