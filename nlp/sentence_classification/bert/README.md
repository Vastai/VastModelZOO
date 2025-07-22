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
- 基于GLUE数据集，BERT模型对下游任务的性能验证

|Model|Score|CoLA|SST-2|MRPC|STS-B|QQP|MNLI-m|MNLI-mm|QNLI(v2)|RTE|WNLI|AX|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|BERT-Tiny|64.2|0.0|83.2|81.1/71.1|74.3/73.6|62.2/83.4|70.2|70.3|81.5|57.2|62.3|21.0|
|BERT-Mini|65.8|0.0|85.9|81.1/71.8|75.4/73.3|66.4/86.2|74.8|74.3|84.1|57.9|62.3|26.1|
|BERT-Small|71.2|27.8|89.7|83.4/76.2|78.8/77.0|68.1/87.0|77.6|77.0|86.4|61.8|62.3|28.6|
|BERT-Medium|73.5|38.0|89.6|86.6/81.6|80.4/78.4|69.6/87.9|80.0|79.1|87.7|62.2|62.3|30.5|

</br>


### 测评数据集说明
####  1. MRPC
[MRPC](https://gluebenchmark.com/) (The Microsoft Research Paraphrase Corpus，微软研究院释义语料库)，相似性和释义任务，是从在线新闻源中自动抽取句子对语料库，并人工注释句子对中的句子是否在语义上等效。类别并不平衡，其中68%的正样本，所以遵循常规的做法，报告准确率（accuracy）和F1值。

- 样本个数：训练集3668个，验证集408个，测试集1725个。
- 任务：句子分类任务，是否释义二分类，是释义，不是释义两类。
- 评价准则：准确率（accuracy）和F1值。

本任务的数据集，包含两句话，每个样本的句子长度都非常长，且数据不均衡，正样本占比68%，负样本仅占32%。

</br>

#### 2. TNEWS
[TNEWS](https://storage.googleapis.com/chineseglue/tasks/tnews.zip)(Short Text Classificaiton for News, 今日头条中文新闻短文本分类)。该数据集来自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游，教育，金融，军事等。
- 样本个数：train 267881条，test 57404条，dev 57403条
- 任务：多分类
- 评价准则：准确率（Accuracy）

</br>

#### 3. IMDB
[IMDB](http://ai.stanford.edu/~amaas/data/sentiment/)是一个大型电影评论数据集，数据集类别为两类，包括积极和消极两种。共提供了一组 2.5万条流行通用的电影评论用于训练，以及2.5万条用于测试。还有其他未标记的数据可供使用。
- 样本个数：train 25000个， test 25000个， unsupervised 50000个
- 任务：二分类（积极、消极）
- 评价准则：准确率（Accuracy）

</br>

#### 4. SST2
[SST2]()，单句子分类任务，包含电影评论中的句子和它们情感的人类注释。这项任务是给定句子的情感，类别分为两类正面情感（positive，样本标签对应为1）和负面情感（negative，样本标签对应为0），并且只用句子级别的标签。也就是，本任务也是一个二分类任务，针对句子级别，分为正面和负面情感。
- 样本个数：训练集67350个，开发集873个，测试集1821个。
- 任务：情感分类，正面情感和负面情感二分类。
- 评价准则：accuracy。

</br>

#### 5. JD
[JD]()，单句子分类任务(情感分类)，该数据集包含从2011年1月1日到2014年3月31日（3年多）某电商网站的消费者购买行为，用户评分，评论和产品元数据，涵盖15个一级产品类别，987个二级产品类别，近2个百万用户，超过10万种产品和超过6,000万条评论。该数据集中的每个文本评论都包含三个子评论：正面评论，负面评论和整体评论。
- 任务：情感分类，正面情感和负面情感二分类。
- 评价准则：accuracy。

</br>

## Build_In Deploy
按照数据集进行 Deploy：
- MRPC
  - [tensorflow&huggingface_deploy](./huggingface/tensorflow_huggingface_mrpc_deploy.md)

- IMDB
  - [huggingface_deploy](./huggingface/huggingface_imdb_deploy.md)

- SST2
  - [huggingface_sst2_telecom_deploy](./huggingface/huggingface_sst2_telecom_deploy.md)

- JD
  - [modelscope_jd_deploy](./modelscope/modelscope_deploy.md)

