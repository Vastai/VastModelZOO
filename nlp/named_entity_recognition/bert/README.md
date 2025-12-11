# **BERT**
[Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962)

## **Model Arch**

BERT是2018年10月由Google AI研究院提出的一种预训练模型。BERT的全称是Bidirectional Encoder Representation from Transformers。其采用了 Transformer 架构的编码器部分用于学习词在给定上下文下词的 Embedding 表示，相较于之前的RNN，其最大的优势是可以并行训练。 BERT 的主要创新点在于pre-train方法上，即用了Masked  LM 和 Next Sentence Prediction 两种方法分别捕捉词语和句子级别的 representation 。因此，使用BERT训练的pre-train 模型对于 NLP 下游任务非常友好。

### Embedding

BERT的 Embedding 处理由三种 Embedding 求和而成：
<div align=center><img src="../../../images/nlp/sentence_classification/bert/bert_token.png"></div>

其中：
- Token Embeddings是词向量，第一个单词是CLS标志，可以用于之后的分类任务
- Segment Embeddings用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务
- Position Embeddings和之前文章中的Transformer不一样，不是三角函数而是学习出来的，而是类似普通的词嵌入一样为每一个位置初始化了一个向量，然后随着网络一起训练得到

</br>

### BertEncoder

如下图所示便是 Bert Encoder 的结构示意图，其整体由多个 BertLayer（也就是论文中所指代的 Transformer blocks）所构成

<div align=center><img src="../../../images/nlp/sentence_classification/bert/bert_backbone.png"></div>

具体的，在论文中作者分别用 L 来表示 BertLayer 的层数，即 BertEncoder 是由 L 个 BertLayer 所构成；用 H 来表示模型的维度；用 A 来表示多头注意力中多 头的个数。同时，在论文中作者分别就 $`BERT_{BASE}`$ (L=12, H=768, A=12) 和 $`BERT_{LARGE}`$ (L=24, H=1024, A=16) 这两种尺寸的 BERT 模型进行了实验对比。 

</br>

### MLM 与 NSP 任务

为了能够更好训练 BERT 网络，论文作者在 BERT 的训练过程中引入两个任务，MLM 和 NSP。对于 MLM 任务来说，其做法是随机掩盖掉输入序列中 15% 的 Token（即用 “[MASK]” 替换掉原有的 Token），然后在 BERT 的输出结果中取对应掩盖位置上的向量进行真实值预测。虽然 MLM 的这种做法能够得到一个很好的预训练模型，但是仍旧存在不足之处。由于在 fine-tuning 时，由于输入序列中并不存在“[MASK]” 这样的 Token，因此这将导致 pre-training 和 fine-tuning 之间存在不匹配不一致的问题（GAP）。为了解决这一问题，作者在原始 MLM 的基础了做了部分改动，即先选定15%的 Token，然后将其中的80%替换为“[MASK]”、10%随机替换为其它 Token、剩下的 10% 不变。最后取这 15% 的 Token 对应的输出做分类来预测其真实值。

由于很多下游任务需要依赖于分析两句话之间的关系来进行建模，例如问题回答等。为了使得模型能够具备有这样的能力，作者在论文中又提出了二分类的下句预测任务具体地，对于每个样本来说都是由 A 和 B 两句话构成，其中 50% 的情况 B 确实为 A 的下一句话（标签为 IsNext），另外的 50% 的情况是 B 为语料中其它 的随机句子（标签为 NotNext），然后模型来预测 B 是否为 A 的下一句话。

<div align=center><img src="../../../images/nlp/sentence_classification/bert/LML_NSP_task_network.png"></div>

如上图所示便是 ML 和 NSP 这两个任务在 BERT 预训练时的输入输出示意图，其中最上层输出的C在预训练时用于 NSP 中的分类任务；其它位置上的 $`T_{i}`$ , $`T^{'}_{j}`$ 则用于预测被掩盖的 Token。

</br>
</br>

## **Model Info**

### 测评数据集说明
####  1. NER (peoples_daily_ner)

对于任意一个NLP任务来说模型最后所要完成的基本上都可看作是一个分类任务。根据给出的标签来看，对于原始句子中的每个字符来说其都有一个对应的类别标签，因此对于NER任务来说只需要对原始句子里的每个字符进行分类即可，然后再将预测后的结果进行后处理便能够得到句子从存在的相应实体。
<div align=center><img src="../../../images/nlp/sentence_classification/bert/ner.png"></div>

[peoples_daily_ner](https://huggingface.co/datasets/peoples_daily_ner) 数据集共有7类，B-表示该类实体的开始标志，I-表示该类实体的延续标志，分别是：

- "B-ORG":组织或公司(organization)
- "I-ORG":组织或公司
- "B-PER":人名(person)
- "I-PER":人名
- "O":其他非实体(other)
- "B-LOC":地名(location)
- "I-LOC":地名

验证集样本：4636个；评价指标：F1和Accuracy。

</br>


## Build_In Deploy

### step.1 模型 finetune
-  huggingface，模型微调说明：[huggingface_bert_peple_daily.md](./huggingface/source_code/finetune/huggingface_bert_peple_daily.md)

### step.2 获取模型
- huggingface，预训练模型导出至torchscript格式，说明: [pt2torchscript](./huggingface/source_code/pretrain_model/README.md)


### step.3 获取数据集
- 校准数据集
    - [huggingface](https://drive.google.com/drive/folders/1nWpOWQrtsB_g9y8JeEMbb71LddHxEUFX)
- [评估数据集-v1.3](https://drive.google.com/drive/folders/1Hj9sk3scunvGQrd48QKOsmOpJINrosMt)
- [评估数据集-v1.5](https://drive.google.com/drive/folders/1QOLVh7TZSrKpm6c34hTGF5wh2OBPRlXm)
- labels: [instance_Peoples_Daily.txt](https://drive.google.com/drive/folders/15O2QrYHCXVP8NjwwByFQ6lUPbYOHvf-r)
> 注意： 由于 compiler 1.5 版本将 bert 相关模型的输入改变为6个。 因此，1.5 版本的校验数据集需使用 `评估数据集-v1.5`。

### step.4 模型转换
1. 根据具体模型修改配置文件
    - [huggingface](./huggingface/build_in/build/huggingface_bert_ner.yaml)

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd bert
    mkdir workspace
    cd workspace
    vamc compile ../huggingface/build_in/build/huggingface_bert_ner.yaml
    vamc compile ../huggingface/build_in/build/huggingface_bert_ner_fp16.yaml
    ```

### step.5 模型推理
- 基于 [sequence2npz.py](../common/utils/sequence2npz.py)，获得推理数据`npz`以及对应的`npz_datalist.txt`

   ```bash
   python ../../common/utils/sequence2npz.py \
       --npz_path /path/to/china-people-daily-ner-corpus/NER256/test4636_6inputs \
       --save_path npz_datalist.txt
   ```

- 推理 运行
  - `compiler version <= 1.5.0 并且 vastsream sdk == 1.X`

    运行 [sample_nlp.py](../common/sdk1.0/sample_nlp.py) 脚本，获取 推理 结果，示例：

    ```bash
    cd ../../common/sdk1.0
    python sample_nlp.py \
        --model_info ./network.json \
        --bytes_size 1024 \
        --datalist_path npz_datalist.txt \
        --save_dir ./output
    ```

    > 可参考 [network.json](../../question_answering/common/sdk1.0/network.json) 进行修改

  - `compiler version >= 1.5.2 并且 vastsream sdk == 2.X`

    运行 [vsx_ner.py](../common/vsx/python/vsx_ner.py) 脚本，获取 推理 结果，示例：

    ```bash
    python ../../common/vsx/python/vsx_ner.py \
        --data_list npz_datalist.txt \
        --model_prefix_path ./deploy_weights/bert_base_chinese_ner/mod \
        --device_id 0 \
        --batch 1 \
        --save_dir ./bert_out

    # run torchscript
    python ../../common/utils/run_torchscript.py \
        --data_list npz_datalist.txt \
        --model_path ../huggingface/source_code/pretrain_model/bert_base_zh_ner-256.torchscript.pt \
        --save_dir ./bert_torch_out
    ```

    > `--model_prefix_path` 为转换的模型三件套的文件前缀路径

- 精度评估

   基于 [people_daily_eval.py](../common/eval/people_daily_eval.py)，解析npz结果，并评估精度
   ```bash

    python ../../common/eval/people_daily_eval.py \
       --result_dir ./bert_out \
       --label_path /path/to/people_daily/instance_Peoples_Daily.txt
   ```

   ```
    # vacc int8
    F1:  0.9024390243902438
    accuracy:  0.9758434270854719

    # vacc fp16
    F1:  0.9590361445783132
    accuracy:  0.9874769041264627

    # torchscript fp32
    F1:  0.9590361445783132
    accuracy:  0.9874769041264627
   ```

### step.6 性能精度测试
1. 基于[sequence2npz.py](../../sentence_classification/common/utils/sequence2npz.py)，生成推理数据`npz`以及对应的`npz_datalist.txt`, 可参考 step.5

2. 执行性能测试：
    ```bash        
    export VSX_DISABLE_DEEPBIND=1
    vamp -m deploy_weights/bert_base_chinese_ner/mod \
        --shape [[1,256],[1,256],[1,256],[1,256],[1,256],[1,256]] \
        --vdsp_params ../../common/vamp_info/bert_vdsp.json \
        --batch_size 1 \
        --instance 1 \
        --processes 1 \
        --datalist npz_datalist.txt  \
        --path_output ./vamp_bert_out
    ```
    > 相应的 `vdsp_params` 等配置文件可在 [vamp_info](../common/vamp_info/) 目录下找到
    >
    > 如果仅测试模型性能可不设置 `datalist`、`path_output` 参数

3. 精度评估
    基于 [people_daily_eval.py](../common/eval/people_daily_eval.py)，解析npz结果，并评估结果，可参考 step.5
    ```bash
    python ../../common/eval/people_daily_eval.py \
       --result_dir ./vamp_bert_out \
       --label_path /path/to/people_daily/instance_Peoples_Daily.txt
   ```