# **XLM-RoBERTa**
[Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)


## **Model Arch**
XLM-RoBERTa 的改进

1. 在 XLM 和 RoBERTa 中使用的跨语言方法的基础上（本质就是 XLM+RoBERTa），在新模型中增加了语种数量和训练数据集的数量，具体来说使用超过 2TB 预处理过的 CommonCrawl 数据集，以无监督的方式训练跨语言表征

2. 在 fine-tuning 期间，基于多语言模型的能力来使用多语言的标注数据，以提升下游任务的性能

3. 调整了模型的参数，以抵消以下不利因素：使用跨语言迁移来将模型扩展到更多的语言时限制了模型理解每种语言的能力。参数更改包括在训练和词汇构建过程中对低资源语言进行上采样，生成更大的共享词汇表，以及将整体模型增加到 5.5 亿参数量。

4. XLM-RoBERTa 的模型主体还是 Transformer，训练目标是多语种的 MLM，基本和 XLM 一样。作者从每个语种的语料中采样出文本，再预测出被 Mask 的 tokens。从各语种采样的方法与 XLM 中相同。另一个与 XLM 不同的是，文本不使用  Language Embeddings。

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
-  huggingface，模型微调说明：[huggingface_xlmroberta_mrpc.md](./huggingface/source_code/finetune/huggingface_xlmroberta_mrpc.md)

### step.2 获取模型
- huggingface，预训练模型导出至torchscript格式，说明: [pt2torchscript](./huggingface/source_code/pretrain_model/README.md)


### step.3 获取数据集
- 校准数据集
    - [huggingface](https://drive.google.com/drive/folders/1FbQr7IYiFJJlY2kCytxYjAkdDLOAJ5Z8)
- [评估数据集-v1.3](https://drive.google.com/drive/folders/1i5iWGYYnfM9LWOoxxer8041iNpbBHVhl)
- [评估数据集-v1.5](https://drive.google.com/drive/folders/1whjFLfxYUjPFOM_ALp17WbfnUTAecDhC)
- labels： [dev.tsv](https://drive.google.com/drive/folders/1kv675JT_IzanhIvB6kiz5_pNFiRdHEFX)
> 注意： 由于 compiler 1.5 版本将 bert 相关模型的输入改变为6个。 因此，1.5 版本的校验数据集需使用 `评估数据集-v1.5`。

### step.4 模型转换
1. 根据具体模型修改配置文件
   - [huggingface](./huggingface/build_in/build/huggingface_xlmroberta_cls.yaml)

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd xlmroberta
    mkdir workspace
    cd workspace
    vamc build ../huggingface/build_in/build/huggingface_xlmroberta_cls.yaml
    ```


### step.5 模型推理

- 基于 [sequence2npz.py](../common/utils/sequence2npz.py)，获得推理数据 `npz` 以及对应的 `npz_datalist.txt`

   ```bash
   python ../../common/utils/sequence2npz.py \
       --npz_path /path/to/MRPC/dev408_6inputs \
       --save_path npz_datalist.txt
   ```

- 推理 运行
  - `compiler version <= 1.5.0 并且 vastsream sdk == 1.X`

    运行 [sample_nlp.py](../common/sdk1.0/sample_nlp.py) 脚本，获取 推理 结果，示例：

    ```bash
    cd ../../common/sdk1.0/
    python sample_nlp.py \
        --model_info ./network.json \
        --bytes_size 512 \
        --datalist_path npz_datalist.txt \
        --save_dir ./result/dev408
    ```

    > 可参考 [network.json](../../question_answering/common/sdk1.0/network.json) 进行修改

  - `compiler version >= 1.5.2 并且 vastsream sdk == 2.X`

    运行 [vsx_sc.py](../common/vsx/python/vsx_sc.py) 脚本，获取 推理 结果，示例：

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

2. 执行测试：
    ```bash
   vamp -m deploy_weights/xlmroberta_base_mrpc-int8-max-mutil_input-vacc/xlmroberta_base_mrpc \
        --vdsp_params ../../common/vamp_info/bert_vdsp.yaml \
        --iterations 1024 \
        --batch_size 1 \
        --instance 6 \
        --processes 2 \
        --datalist npz_datalist.txt \
        --path_output ./save/xlmroberta
    ```
    > 相应的 `vdsp_params` 等配置文件可在 [vamp_info](../common/vamp_info/) 目录下找到
    >
    > 如果仅测试模型性能可不设置 `datalist`、`path_output` 参数

3. 精度评估

    基于[mrpc_eval.py](../common/eval/mrpc_eval.py)，解析和评估npz结果， 可参考 step.5
 
