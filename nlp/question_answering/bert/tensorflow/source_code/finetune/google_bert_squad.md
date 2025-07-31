# **finetune**

## **Bert Squad任务微调**

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
- 从[Squad 网站](https://rajpurkar.github.io/SQuAD-explorer/)下载squad-1.1 数据集，到 `bert/squad` 目录下。

### **step.3 预训练模型准备**
- 根据官网提供的预训练模型下载地址，下载 [bert_base_uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) 预训练模型, 并解压放置至 `bert/weights` 目录下。

### **step.4  模型微调**
待数据集与预训练模型准备完毕后，开始进行模型微调：

- 添加数据集和预训练模型环境变量
  ```bash
  export BERT_BASE_DIR=/path/to/bert/weights/uncased_L-12_H-768_A-12  # 预训练模型路径
  
  export SQUAD_DIR=/path/to/bert/datasets/squad    # squad-1.1 数据集路径
  ```

- 运行 run_classifier.py, 微调模型：
  ```bash
  python run_squad.py \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --do_train=True \
    --train_file=$SQUAD_DIR/train-v1.1.json \
    --do_predict=True \
    --predict_file=$SQUAD_DIR/dev-v1.1.json \
    --train_batch_size=12 \
    --learning_rate=3e-5 \
    --num_train_epochs=2.0 \
    --max_seq_length=384 \
    --doc_stride=128 \
    --output_dir=/tmp/squad_base/
  ```

- 待微调结束后会在控制台得到类似这样的输出信息：

  ```
  {"f1": 88.41249612335034, "exact_match": 81.2488174077578}
  ```
  > 微调后，F1 精度在 85 ~ 89 之间均属于正常范围。

- 模型训练权重ckpt文件，保存至 `/path/to/bert/tmp/squad_output` 目录，以备后续模型量化编译使用。