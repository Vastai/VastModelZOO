# **finetune**

## **Bert Squad任务微调**

### **step.1  环境准备**
- 本地安装训练环境：
  ```bash
  conda create -n hf_bert python=3.7
  conda activate hf_bert
  cd /path/to/vastmodelzoo/question_answering/bert/huggingface/source_code/finetune
  pip install -r requirements.txt  #
  ```

### **step.2  模型微调**

- 运行 shell 脚本, 微调模型：

  ```bash
  python run_qa.py \
    --model_name_or_path bert-base-cased \    # 模型名，如果未提前下载，则会重新下载
    --dataset_name squad \      # 数据集名，如果未提前下载，则会重新下载
    --do_train \                # 是否进行训练
    --do_eval \                 # 是否进行精度验证
    --per_device_train_batch_size 12 \ 
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \      # sequence legnth
    --doc_stride 128 \
    --output_dir ./code/nlp/transformers/runs/SQuAD_for_qa_bert_base_uncased_384    # 输出路径
  ```

- 模型微调训练会每隔 500 次 iter 进行保存，选择其中精度最高的预训练权重，以备后续模型量化编译使用。