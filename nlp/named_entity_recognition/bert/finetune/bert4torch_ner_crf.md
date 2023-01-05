# **Bert - finetune**

## **Bert NER 任务微调**

### **step.1  环境准备**

- 通过 github 下载 tensorflow bert 官方源码，并搭建 bert 训练环境：
  ```bash
  conda create -n torch_bert python=3.7
  conda activate torch_bert
  git clone https://github.com/Tongjilibo/bert4torch.git
  cd bert4torch
  pip install -r requirements.txt  # 推荐在torch 1.10 版本下进行训练
  ```

### **step.2 数据集准备**

- 根据[源仓库](https://github.com/Tongjilibo/bert4torch/tree/master/examples)提供的数据下载地址，下载中文命名实体识别数据集 [china-people-daily-ner-corpus](http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz) , 并放置到 `path/to/bert4torch/datasets` 目录下。

### **step.3 预训练模型准备**

- 根据官网提供的预训练模型下载地址， 下载 [bert_base_chinese](https://huggingface.co/bert-base-chinese/blob/main/pytorch_model.bin) 预训练模型, 并解压放置至 `path/to/bert4torch/weights/bert_base_chinese` 目录下;
- 下载相对应的 [config](https://huggingface.co/bert-base-chinese/resolve/main/config.json) 和 [vocab](https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt) 文件， 放置 `path/to/bert4torch/weights/bert_base_chinese` 目录下;

### **step.4  模型微调**
- 模型权重转换:
    - 使用 bert4torch 中的 convert_bert-base-chinese.py 对 bert 中文预训练模型权重进行转换:
        ```bash
        cd /path/to/bert4torch/examples/convert_script
        vim convert_bert-base-chinese.py
        ```
    - 替换模型路径和保存路径：
        ```python
        import torch

        state_dict = torch.load('../../weights/bert_base_chinese/pytorch_model.bin')  # 替换成下载的中文预训练模型权重路径
        state_dict_new = {}
        for k, v in state_dict.items():
            if 'LayerNorm.gamma' in k:
                k = k.replace('LayerNorm.gamma', 'LayerNorm.weight')
                state_dict_new[k] = v
            elif 'LayerNorm.beta' in k:
                k = k.replace('LayerNorm.beta', 'LayerNorm.bias')
                state_dict_new[k] = v
            else:
                state_dict_new[k] = v
        torch.save(state_dict_new, '../../weights/bert_base_chinese/pytorch_model_cvt.bin')  # 保存转换的模型路径
        ```
- 模型微调训练
    - 编辑 task_sequence_labeling_ner_crf.py 代码:
        ```bash
        cd /path/to/bert4torch/examples/sequence_labeling/
        vim task_sequence_labeling_ner_crf.py
        ```

    - 从代码中替换 config_path（配置文件）、 checkpoint_path（转换模型权重）、 dict_path（vocab）的路径，替换 train_dataloader、valid_dataloader的数据集路径
        ```python
        # BERT base
        config_path = '../../weights/bert_base_chinese/config.json'
        checkpoint_path = '../../weights/bert_base_chinese/pytorch_model_cvt.bin'
        dict_path = '../../weights/bert_base_chinese/vocab.txt'

        # 转换数据集
        train_dataloader = DataLoader(MyDataset('../../datasets/china-people-daily-ner-corpus/example.train'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
        valid_dataloader = DataLoader(MyDataset('./dataset//china-people-daily-ner-corpus/example.dev'), batch_size=batch_size, collate_fn=collate_fn) 
        ```
    
    - 运行 task_sequence_labeling_ner_crf.py
        ```bash
        python task_sequence_labeling_ner_crf.py
        ```
    - 待脚本运行结束后， 可得到finetune模型，best_model.pt， 为后续模型量化编译准备。