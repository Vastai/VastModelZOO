# **finetune**

## **Bert Squad任务微调**


- 运行 shell 脚本, 微调模型：

    ```bash
    export TRANSFORMERS_OFFLINE=1
    export SQUAD_DIR=/path/to/datasets/squad

    python run_qa.py \
    --model_name_or_path pretrain/huggingface/ernie2.0-large-en \
    --do_train \
    --do_eval \
    --dataset_name $SQUAD_DIR \
    --per_device_train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir finetune/qa/ernie2.0-large-en \
    --overwrite_output_dir
    ```

- 模型微调训练会每隔 500 次 iter 进行保存，选择其中精度最高的预训练权重，以备后续模型量化编译使用。