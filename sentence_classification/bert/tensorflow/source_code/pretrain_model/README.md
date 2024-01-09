# Convert Pretrain Model

将ckpt2pb_classifer.py 移动至 [step.1](../finetune/google_bert_mrpc.md) 使用的 tensorflow bert repo 目录下，运行 ckpt2pb_classifer.py
```bash
cd path/to/bert/

python ckpt2pb_classifer.py \
    --init_checkpoint path/to/bert/tmp/mrpc_output/model.ckpt \
    --bert_config_file path/to/bert/weights/uncased_L-12_H-768_A-12/bert_config.json \
    --output_dir output/google_bert_mrpc
```
- init_checkpoint：根据 step.1 进行微调后的 `ckpt` 权重文件路径
- bert_config_file： 预训练模型 bert-base_uncased 的配置文件路径
- output_dir： 模型转换后的 `pb` 格式文件保存路径