# Convert Pretrain Model

将ckpt2pb_qa.py 移动至 [step.1](../finetune/google_bert_squad.md) 使用的 tensorflow bert repo 目录下，运行 ckpt2pb_qa.py
```bash
cd path/to/bert/

python ckpt2pb_qa.py  \ 
    --bert_config_file ./bert_base_uncased/bert_config.json  \
    --task_name squad  \
    --init_checkpoint path/to/tmp/bert_base_squad/model.ckpt-14599 \
    --output_dir path/to/out_dir
```
- init_checkpoint：根据 step.1 进行微调后的 `ckpt` 权重文件路径
- bert_config_file： 预训练模型 bert-base_uncased 的配置文件路径
- output_dir： 模型转换后的 `pb` 格式文件保存路径