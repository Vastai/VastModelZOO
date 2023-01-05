# **BERT**
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## **Model Arch**
[ModelArch](../../sentence_classification/bert/README.md)

<br/>

## **测评数据集说明**
### china-people-daily-ner-corpus
对于任意一个NLP任务来说模型最后所要完成的基本上都可看作是一个分类任务。根据给出的标签来看，对于原始句子中的每个字符来说其都有一个对应的类别标签，因此对于NER任务来说只需要对原始句子里的每个字符进行分类即可，然后再将预测后的结果进行后处理便能够得到句子从存在的相应实体。

china-people-daily-ner-corpus 是一个中文命名实体识别数据集，共有7类，其中 B- 表示该类实体的开始标志，I- 表示该类实体的延续标志，分别是：

- "B-ORG":组织或公司(organization)
- "I-ORG":组织或公司
- "B-PER":人名(person)
- "I-PER":人名
- "O":其他非实体(other)
- "B-LOC":地名(location)
- "I-LOC":地名
测试集样本：4646个；评价指标：F1和Accuracy。

<br/>

## **VACC**

### step.1 模型 finetune 
- bert4torch  模型微调说明：[bert4torch_ner_crf](./finetune/bert4torch_ner_crf.md)

### step.2 准备预训练模型
- bert4torch: 从 step.1 bert4torch 获得预训练模型；

### step.3 准备数据集
- china-people-daily-ner-corpus
    - china-people-daily-ner-corpus 数据集为 bert ner 任务微调使用的公开数据集， 用户可自行准备数据集, [china-people-daily](http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz)
        ```
        ├── china-people-daily-ner-corpus
        |   ├── example.train  
        │   ├── example.dev
        |   ├── example.test
        |   ├── .....
        ```
    - 模型量化以及推理所需数据的格式为`.npz`，输入为3个, 包括：`inputs_ids、segment_ids、input_mask`，输入长度为256，数据类型为 `int32` , 用户可依据上述要求自行解析 china-people-daily-ner-corpus 数据集。

### step.4 模型转换

 - 根据具体模型修改配置文件: [bert4torch_ner](./build_config/bert4torch_ner.yaml)

 - 命令行执行转换:
   ```bash
   vamc build ./build_config/bert4torch_ner.yaml
   ```

### step.5 模型推理和精度评估
- 根据step.4 配置模型三件套信息，[model_info](./model_info/model_info_bert_ner.json)
- 执行推理，调用入口 [sample_bert_ner](../../../inference/nlp/named_entity_recognition/bert/sample_bert_ner.py)， 源码可参考 [nlp_bert](../../../inference/nlp/utils/nlp_bert.py)
    ```bash
    # 执行run stream 脚本
    cd ../../inference/nlp/bert

    python sample_bert_ner.py \
      --task_name ner \
      --model_info /path/to/nlp/bert/model_info/model_info_bert_ner.json \
      --eval_path /path/to/data/china_people_daily/example.test \
      --bytes_size 1024 \
      --data_dir /path/to/data/china_people_daily/test4636 \
      --save_dir /path/to/output
    ```
    - task_name： nlp下游任务名称
    - model_info：指定上阶段转换的模型路径
    - eval_path：精度评估数据集的路径
    - bytes_size： 转换成bytes的数据大小，如128长度的序列，数据类型为int32，则转换成bytes类型后的长度为 256 * 4 = 1024
    - data_dir： step.3 阶段转换成npz格式的数据，用于模型推理所需数据
    - save_dir： 输出 feature 和精度评估结果保存路径地址

-  在执行 `sample_bert_ner.py` 脚本后， 会在模型推理阶段结束后进行精度评估，并保存精度评估结果



