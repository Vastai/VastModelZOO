# **finetune**

## **Bert MRPC任务微调**

### **step.1  环境准备**
- 通过 github 下载 tensorflow bert 官方源码，并搭建 bert 训练环境：
  ```bash
  conda create -n bert python=3.7
  conda activate bert
  git clone https://github.com/google-research/bert.git
  cd bert
  pip install -r requirements.txt  # 推荐在tensorflow 1.x 环境下进行训练
  ```

### **step.2 数据集准备**
- 从[GLUE网站](https://gluebenchmark.com/)下载MRPC数据集，到 `bert/glue` 目录下。

### **step.3 预训练模型准备**
- 根据官网提供的预训练模型下载地址，下载 [bert_base_uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) 预训练模型, 并解压放置至 `bert/weights` 目录下。

### **step.4  模型微调**
待数据集与预训练模型准备完毕后，开始进行模型微调：

- 添加数据集和预训练模型环境变量
  ```bash
  export BERT_BASE_DIR=/path/to/bert/weights/uncased_L-12_H-768_A-12  # 预训练模型路径
  
  export GLUE_DIR=/path/to/bert/glue/MRPC    # MRPC数据集路径
  ```

- 运行 run_classifier.py, 微调模型：
  ```bash
  python run_classifier.py \
    --task_name=MRPC \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=tmp/mrpc_output/
  ```

- 待微调结束后会在控制台得到类似这样的输出信息：

  ```
  ***** Eval results *****
    eval_accuracy = 0.845588
    eval_loss = 0.505248
    global_step = 343
    loss = 0.505248
  ```
  > 由于MRPC数据集较小，模型微调波动较大，微调的精度范围在84% ~ 89%之间均属于正常。

- 模型训练权重ckpt文件，保存至 `/path/to/bert/tmp/mrpc_output` 目录，以备后续模型量化编译使用。