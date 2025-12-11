# **RoBERTa**
[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

## **Model Arch**

RoBERTa 是 2019 年由华盛顿大学与 Meta 联合发表的， RoBERTa 是 BERT 的改进版，它在模型规模、算力和数据上，与 BERT 相比主要有以下几点改进：
- 更大的模型参数量（论文提供的训练时间来看，模型使用 1024 块 V100 GPU 训练了 1 天的时间）
- 更大bacth size，RoBERTa 在训练过程中使用了更大的 bacth size。尝试过从 256 到 8000 不等的 bacth size
- 更多的训练数据（包括：CC-NEWS 等在内的 160GB 纯文本。而最初的 BERT 使用 16GB BookCorpus 数据集和英语维基百科进行训练）

另外，RoBERTa 在训练方法上有以下改进：

- 去掉下一句预测(NSP)任务
- 动态掩码。BERT 依赖随机掩码和预测 token。原版的 BERT 实现在数据预处理期间执行一次掩码，得到一个静态掩码。 而 RoBERTa 使用了动态掩码：每次向模型输入一个序列时都会生成新的掩码模式。这样，在大量数据不断输入的过程中，模型会逐渐适应不同的掩码策略，学习不同的语言表征。
- 文本编码。Byte-Pair Encoding（BPE）是字符级和词级别表征的混合，支持处理自然语言语料库中的众多常见词汇。原版的 BERT 实现使用字符级别的 BPE 词汇，大小为 30K，是在利用启发式分词规则对输入进行预处理之后学得的。Facebook 研究者没有s采用这种方式，而是考虑用更大的 byte 级别 BPE 词汇表来训练 BERT，这一词汇表包含 50K 的 subword 单元，且没有对输入作任何额外的预处理或分词。

RoBERTa建立在BERT的语言掩蔽策略的基础上，修改BERT中的关键超参数，包括删除BERT的下一个句子训练前目标，以及使用更大的bacth size和学习率进行训练。RoBERTa也接受了比BERT多一个数量级的训练，时间更长。这使得RoBERTa表示能够比BERT更好地推广到下游任务。
<div align=center><img src="../../../images/nlp/sentence_classification/roberta/roberta.jpg" width='80%'></div>
</br>

### DATA

RoBERTa 采用 160 G 训练文本，远超 BERT 的 16G 文本，其中包括：

- BOOKCORPUS 和英文维基百科：原始 BERT 的训练集，大小 16GB。
- CC-NEWS：包含2016年9月到2019年2月爬取的6300万篇英文新闻，大小 76 GB（经过过滤之后）。
- OPENWEBTEXT：从 Reddit 上共享的 URL （至少3个点赞）中提取的网页内容，大小 38 GB 。
- STORIES：CommonCrawl 数据集的一个子集，包含 Winograd 模式的故事风格，大小 31GB 。

</br>

### Dynamic Masking
- 原始静态 mask：
BERT中是准备训练数据时，每个样本只会进行一次随机mask（因此每个epoch都是重复），后续的每个训练步都采用相同的mask，这是原始静态mask，即单个静态mask，这是原始 BERT 的做法。

- 修改版静态 mask：
在预处理的时候将数据集拷贝 10 次，每次拷贝采用不同的 mask（总共40 epochs，所以每一个mask对应的数据被训练4个epoch）。这等价于原始的数据集采用10种静态 mask 来训练 40个 epoch。

- 动态 mask：
并没有在预处理的时候执行 mask，而是在每次向模型提供输入时动态生成 mask，所以是时刻变化的。


</br>

### Model Input Format and NSP

原始的 BERT 包含 2 个任务，预测被 mask 掉的单词和下一句预测。鉴于最近有研究开始质疑下一句预测(NSP)的必要性，本文设计了以下4种训练方式：

- **SEGMENT-PAIR + NSP：**
输入包含两部分，每个部分是来自同一文档或者不同文档的 segment （segment 是连续的多个句子），这两个segment 的token总数少于 512 。预训练包含 MLM 任务和 NSP 任务。这是原始 BERT 的做法。

- **SENTENCE-PAIR + NSP：**
输入也是包含两部分，每个部分是来自同一个文档或者不同文档的单个句子，这两个句子的token 总数少于 512 。由于这些输入明显少于512 个tokens，因此增加batch size的大小，以使 tokens 总数保持与SEGMENT-PAIR + NSP 相似。预训练包含 MLM 任务和 NSP 任务。

- **FULL-SENTENCES：**
输入只有一部分（而不是两部分），来自同一个文档或者不同文档的连续多个句子，token 总数不超过 512 。输入可能跨越文档边界，如果跨文档，则在上一个文档末尾添加文档边界token 。预训练不包含 NSP 任务。

- **DOC-SENTENCES：**
输入只有一部分（而不是两部分），输入的构造类似于FULL-SENTENCES，只是不需要跨越文档边界，其输入来自同一个文档的连续句子，token 总数不超过 512 。在文档末尾附近采样的输入可以短于 512个tokens， 因此在这些情况下动态增加batch size大小以达到与 FULL-SENTENCES 相同的tokens总数。预训练不包含 NSP 任务。

</br>

## **Model Info**

### 测评数据集说明
####  1. MRPC
[MRPC](https://gluebenchmark.com/) (The Microsoft Research Paraphrase Corpus，微软研究院释义语料库)，相似性和释义任务，是从在线新闻源中自动抽取句子对语料库，并人工注释句子对中的句子是否在语义上等效。类别并不平衡，其中68%的正样本，所以遵循常规的做法，报告准确率（accuracy）和F1值。

- 样本个数：训练集3668个，验证集408个，测试集1725个。
- 任务：句子分类任务，是否释义二分类，是释义，不是释义两类。
- 评价准则：准确率（accuracy）和F1值。

本任务的数据集，包含两句话，每个样本的句子长度都非常长，且数据不均衡，正样本占比68%，负样本仅占32%。

</br>


## Build_In Deploy

### step.1 模型 finetune
-  huggingface，模型微调说明：[huggingface_roberta_mrpc.md](./source_code/finetune/huggingface_roberta_mrpc.md)

### step.2 获取模型
- huggingface，预训练模型导出至torchscript格式，说明: [pt2torchscript](./source_code/pretrain_model/README.md)


### step.3 获取数据集
- 校准数据集
    - [huggingface](https://drive.google.com/drive/folders/1FbQr7IYiFJJlY2kCytxYjAkdDLOAJ5Z8)
- [评估数据集-v1.3](https://drive.google.com/drive/folders/1i5iWGYYnfM9LWOoxxer8041iNpbBHVhl)
- [评估数据集-v1.5](https://drive.google.com/drive/folders/1whjFLfxYUjPFOM_ALp17WbfnUTAecDhC)
- labels： [dev.tsv](https://drive.google.com/drive/folders/1kv675JT_IzanhIvB6kiz5_pNFiRdHEFX)
> 注意： 由于 compiler 1.5 版本将 bert 相关模型的输入改变为6个。 因此，1.5 版本的校验数据集需使用 `评估数据集-v1.5`。

### step.4 模型转换
1. 根据具体模型修改配置文件
    - [huggingface](./build_in/build/huggingface_roberta_cls.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd roberta
    mkdir workspace
    cd workspace
    vamc build ../build_in/build/huggingface_roberta_cls.yaml
    ```

### step.5 模型推理

- 基于 [sequence2npz.py](../common/utils/sequence2npz.py)，获得推理数据 `npz` 以及对应的 `npz_datalist.txt`

   ```bash
   python ../../common/utils/sequence2npz.py \
       --npz_path /path/to/MRPC/dev408_6inputs \
       --save_path npz_datalist.txt
   ```

- runstream 运行

  - `compiler version >= 1.5.2 并且 vastsream sdk == 2.X`

    运行 [vsx_sc.py](../common/vsx/python/vsx_sc.py) 脚本，获取 runstream 结果，示例：

    ```bash
    cd ../../common/vsx/python/
    python vsx_sc.py \
        --data_list npz_datalist.txt\
        --model_prefix_path ./build_deploy/bert_base_128/bert_base_128 \
        --device_id 0 \
        --batch 1 \
        --save_dir ./result/dev408
    ```

    > `--model_prefix_path` 为转换的模型三件套的文件前缀路径

- 精度评估

   基于[mrpc_eval.py](../common/eval/mrpc_eval.py)，解析npz结果，并评估精度
   ```bash
   python ../../common/eval/mrpc_eval.py --result_dir ./result/dev408 --eval_path /path/to/MRPC/dev.tsv
   ```

### step.6 性能精度测试
1. 基于[sequence2npz.py](../common/utils/sequence2npz.py)，获得推理数据`npz`以及对应的`npz_datalist.txt`， 可参考 step.5

2. 执行性能测试
    ```bash
   vamp -m deploy_weights/roberta_base_mrpc-int8-max-mutil_input-vacc/roberta_base_mrpc \
        --vdsp_params ../../common/vamp_info/bert_vdsp.json \
        --iterations 1024 \
        --batch_size 1 \
        --instance 6 \
        --processes 2 \
        --datalist npz_datalist.txt \
        --path_output ./save/bert
    ```
    > 相应的 `vdsp_params` 等配置文件可在 [vamp_info](../common/vamp_info/) 目录下找到
    >
    > 如果仅测试模型性能可不设置 `datalist`、`path_output` 参数

3. 精度评估

    基于[mrpc_eval.py](../common/eval/mrpc_eval.py)，解析和评估npz结果， 可参考 step.5
 
