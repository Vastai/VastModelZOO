# HuggingFace Bert-cls Finetune

## finetune MRPC 英文序列分类数据集
运行 ernie_cls_mrpc_finetune.py
```bash
cd path/to/modelzoo/vastmodelzoo/sentence_classification/ernie/huggingface/source_code/finetune

python ernie_cls_finetune.py \
    --model_name_or_path path/to/pretrain_model/ernie2.0-base-en \
    --output_dir path/to/output/MRPC_for_seqclassification_ernie2_base_en \
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

</br>

### Tips

如果是下载的 huggingface 网站上的 ERNIE 3.0 版本的模型进行微调，需要在运行`ernie_cls_finetune.py` 之前修改预训练模型里的config.json：
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