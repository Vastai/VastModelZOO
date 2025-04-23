# BGE

- Technical Report
  - https://arxiv.org/abs/2402.03216
- Huggingface
  - https://huggingface.co/BAAI


## Model Arch
BGE 出色的语义表征能力源于两方面要素：
- 针对表征的预训练
- 大规模文本对训练

BGE在悟道 、Pile 两个大规模语料集上采取了针对表征的预训练算法 RetroMAE：将低掩码率的输入编码为语义向量（Embed），再将高掩码率的输入与语义向量拼接以重建原始输入。这样一来，BGE 得以利用无标签语料实现语言模型基座对语义表征任务的适配。

![img](https://raw.githubusercontent.com/Hiwyl/typora/main/imgs/202406200959840.png)

BGE 针对中文、英文分别构建了多达**120M、232M**的样本对数据，从而帮助模型掌握实际场景中各种不同的语义匹配任务，并借助**负采样扩增**与**难负样例挖掘**进一步提升对比学习的难度，实现了多达**65K的负样本规模**，增强了语义向量的判别能力

另外，BGE 借鉴 Instruction Tuning的思想，采取了非对称的指令添加方式，在问题端添加场景描述， 提升了语义向量在多任务场景下的通用能力

![img](https://raw.githubusercontent.com/Hiwyl/typora/main/imgs/202406201748972.webp)



## Embedding

### Training

![BGE-M3采用多阶段](https://raw.githubusercontent.com/Hiwyl/typora/main/imgs/202406201147610.png)

BGE-M3模型训练分为三个阶段：

-  **RetroMAE预训练**，在105种语言的网页数据和wiki数据上进行，提供一个可以支持8192长度和面向表示任务的基座模型；
-  **无监督对比学习**，在194种单语言和1390种翻译对数据共1.1B的文本对上进行的大规模对比学习；
-  **多检索方式统一优化**，在高质量多样化的数据上进行多功能检索优化，使模型具备多种检索能力。

#### 1. 自蒸馏

人类可以利用多种不同的方式计算结果，矫正误差。模型也可以，通过联合多种检索方式的输出，可以取得比单检索模式更好的效果。因此，BGE-M3使用了一种自激励蒸馏方法来提高检索性能。具体来说，合并三种检索模式的输出，得到新的文本相似度分数，将其作为激励信号，让各单模式学习该信号，以提高单检索模式的效果。

![image.png](https://raw.githubusercontent.com/Hiwyl/typora/main/imgs/202406201748118.png)

#### 2. 训练效率优化

通过根据长度对文本数据进行分组，确保一个batch内文本长度相对相似，从而减少填充。为了减少文本建模时的显存消耗，将一批数据分成多个小批。对于每个小批，利用模型编码文本，收集输出的向量同时丢弃所有前向传播中的中间状态，最后汇总向量计算损失，可以显著增加训练的 `batch size`。

![Efficient Batching](https://raw.githubusercontent.com/Hiwyl/typora/main/imgs/202406261556800.png)

#### 3. 长文本优化

BGE-M3提出了一种简单而有效的方法：MCLS(Multiple CLS)来增强模型的能力，而无需对长文本进行微调。

MCLS方法旨在利用多个CLS令牌来联合捕获长文本的语义。为每个固定数量的令牌插入一个cls令牌，每个cls令牌可以从相邻的令牌获取语义信息，最后通过对所有cls令牌的最后隐藏状态求平均值来获得最终的文本嵌入。

![MCLS](https://raw.githubusercontent.com/Hiwyl/typora/main/imgs/202406261557606.png)



### model analysis

- 使用`Embedding Model` 一般不需要 `pooler`模块

#### pre-processing

1. tokenizer

   - attention_mask

   - (optional) token_type_ids

     ```python
     np.zeros(features['input_ids'].shape, dtype=np.int32)
     ```


#### model_info
<details>
 <summary>展开查看</summary>
 <pre><code>   
==============================================================================================================
Layer (type:depth-idx)                                       Output Shape              Param #
==============================================================================================================
XLMRobertaModel                                              [1, 1024]                 --
├─XLMRobertaEmbeddings: 1-1                                  [1, 6, 1024]              --
│    └─Embedding: 2-1                                        [1, 6, 1024]              256,002,048
│    └─Embedding: 2-2                                        [1, 6, 1024]              1,024
│    └─Embedding: 2-3                                        [1, 6, 1024]              8,390,656
│    └─LayerNorm: 2-4                                        [1, 6, 1024]              2,048
│    └─Dropout: 2-5                                          [1, 6, 1024]              --
├─XLMRobertaEncoder: 1-2                                     [1, 6, 1024]              --
│    └─ModuleList: 2-6                                       --                        --
│    │    └─XLMRobertaLayer: 3-1                             [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-2                             [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-3                             [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-4                             [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-5                             [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-6                             [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-7                             [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-8                             [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-9                             [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-10                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-11                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-12                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-13                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-14                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-15                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-16                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-17                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-18                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-19                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-20                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-21                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-22                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-23                            [1, 6, 1024]              12,596,224
│    │    └─XLMRobertaLayer: 3-24                            [1, 6, 1024]              12,596,224
├─XLMRobertaPooler: 1-3                                      [1, 1024]                 --
│    └─Linear: 2-7                                           [1, 1024]                 1,049,600
│    └─Tanh: 2-8                                             [1, 1024]                 --
==============================================================================================================
Total params: 567,754,752
Trainable params: 567,754,752
Non-trainable params: 0
Total mult-adds (M): 567.75
==============================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 13.18
Params size (MB): 2271.02
Estimated Total Size (MB): 2284.20
==============================================================================================================
 </code></pre>
</details>

###### Overview
<details>
 <summary>展开查看</summary>
 <pre><code>  
| Aten Operations  | NN Operations  |
| :--------------: | :------------: |
|    aten::add     |    Dropout     |
|    aten::add_    |   Embedding    |
| aten::contiguous | GELUActivation |
|   aten::cumsum   |   LayerNorm    |
|    aten::div     |     Linear     |
| aten::embedding  |      Tanh      |
|   aten::expand   |                |
|    aten::gelu    |                |
| aten::layer_norm |                |
|   aten::matmul   |                |
|    aten::mul     |                |
|     aten::ne     |                |
|  aten::permute   |                |
|    aten::rsub    |                |
|   aten::select   |                |
|    aten::size    |                |
|   aten::slice    |                |
|  aten::softmax   |                |
|    aten::tanh    |                |
|     aten::to     |                |
| aten::transpose  |                |
|  aten::type_as   |                |
| aten::unsqueeze  |                |
|    aten::view    |                |
 </code></pre>
</details>


#### post-processing

```python
# forward
outputs = model(**inputs_on_device, return_dict=True)
# last_hidden_state / pooler_output
# output.last_hidden_state : [1, x, 1024]
# output.pooler_output : [1, 1024] - 一般不导出该模块
embeddings = outputs.last_hidden_state[:, 0]
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) # [1, 1024]
```

M3-Embedding统一了嵌入模型的三种常见检索功能，即密集检索（Dense retrieval）、词汇（稀疏）检索（Lexical retrieval）和多向量检索（Multi-vector retrieval）。以下是这些方法的公式化描述：

1. 密集检索（Dense retrieval）：输入查询q被转换为基于文本编码器的隐藏状态Hq，使用特殊标记“[CLS]”的归一化隐藏状态来表示查询：查询和段落之间的相关性得分通过两个嵌入向量 ep 和 eq的内积来度量

   ```python
   def dense_embedding(self, hidden_state, mask):
       if self.sentence_pooling_method == 'cls':
           return hidden_state[:, 0]
       elif self.sentence_pooling_method == 'mean':
           s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
           d = mask.sum(axis=1, keepdim=True).float()
           return s / d
   ```

2. 词汇检索（Lexical Retrieval）：输出嵌入还被用来估计每个词项的重要性，以促进词汇检索：对于查询中的每个词项t（在我们的工作中，词项对应于一个标记）如果词项t在查询中出现多次，我们只保留其最大权重。我们以相同的方式计算段落中每个词项的权重。基于估计的词项权重，查询和段落之间的相关性得分通过查询和段落中共同出现的词项的联合重要性计算

   ```python
   def sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = True):
       # sparse_linear 线性层= torch.nn.Linear(in_features=self.model.config.hidden_size, out_features=1)
       # 通过relu计算token weight
       token_weights = torch.relu(self.sparse_linear(hidden_state))
       if not return_embedding: return token_weights
       # 形状为(input_ids.size(0), input_ids.size(1), self.vocab_size)的零张量
       sparse_embedding = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab_size,
                                      dtype=token_weights.dtype,
                                      device=token_weights.device)
       # 将token_weights中的值分散scatter到sparse_embedding的相应位置，索引位置根据input_ids提供
       sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)
       # CLS，PAD 等无用token
       unused_tokens = [self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                        self.tokenizer.unk_token_id]
       sparse_embedding = torch.max(sparse_embedding, dim=1).values
       #  无用token weight设置为0
       sparse_embedding[:, unused_tokens] *= 0.
       return sparse_embedding
   ```

3. 多向量检索（Multi-Vector Retrieval）：作为密集检索的扩展，多向量方法利用整个输出嵌入来表示查询和段落：

## Reranker

1. 前处理同`Embedding`模型，区别在于输入为`List`的文本对

2. 模型结构同 `Embedding`模型，区别在于`pooler`模块换为了`Classifier`模块，结果 shape为 `[bs, 1]`

3. 后处理如下：

   ```python
   output = self.model(**inputs, return_dict=True) # [bs, 1]
   scores = output.logits.view(-1,).float()
   ```

## Build_In Deploy

### step.1 模型准备

1. 下载模型权重

- Embedding
    |                          Model Name                          | Dimension | Sequence Length |                         Introduction                         |
    | :----------------------------------------------------------: | :-------: | :-------------: | :----------------------------------------------------------: |
    |      [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)       |   1024    |      8192       | multilingual; unified fine-tuning (dense, sparse, and colbert) </br> from bge-m3-unsupervised |
    | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |    384    |       512       |                      English model                         |
    | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |    768    |       512       |                        English model                         |
    | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |   1024    |       512       |                      English model                         |
    | [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) |    384    |       512       |                      Chinese model
    | [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |    768    |       512       |                        Chinese model                         |
    | [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) |   1024    |       512       |                      Chinese model                         |
    
    >  base on XLMRobertaModel

- ReRanker

    | 模型 |   基础模型  |   语言   | Dimension | Sequence Length  | Note                                                         |
    | :----------------------------------------------------------: | :----------------------------------------------------------: | :------: | :--: |:--: |:--: | 
    | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)  |  中英文  |  768 | 512 | 轻量级重排序模型，易于部署，推理速度快。                     |
    | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | [xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) |  中英文  |   1024 | 512 | 轻量级重排序模型，易于部署，推理速度快。                     |
    | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |         [bge-m3](https://huggingface.co/BAAI/bge-m3)         | 多种语言 |   1024 | 8192 | 轻量级重排序模型，具有强大的多语言能力，易于部署，推理速度快。 |


2. 模型导出onnx：[onnx_export.py](../common/source_code/onnx_export.py)

    ```bash
    python ../common/source_code/onnx_export.py \
        --model bge/bge-m3 \
        --task embedding \
        --seqlen 512 \
        --save_dir ./onnx_weights
    ```

### step.2 数据集
1. 精度评估数据集：
    - embedding
        - 英文：[mteb/sts12-sts](https://huggingface.co/datasets/mteb/sts12-sts)
        - 中文：[C-MTEB/BQ](https://huggingface.co/datasets/C-MTEB/BQ)
    - reranker：[zyznull/msmarco-passage-ranking](https://huggingface.co/datasets/zyznull/msmarco-passage-ranking)
    - 数据集下载和转换为jsonl格式：[download_datasets.py](../common/source_code/download_datasets.py)
2. 量化数据集：
    - [gen_quant_data.py](../common/source_code/gen_quant_data.py)，基于以上数据集，指定seqlen，合成npz量化数据集

### step.3 模型转换

1. 参考瀚博训推软件生态链文档，获取模型转换工具: [vamc v3.0+](../../docs/vastai_software.md)
2. 根据具体模型修改模型转换配置文件
    - [embedding_config_fp16.yaml](./build_in/build/embedding_config_fp16.yaml)
    - [embedding_config_int8.yaml](./build_in/build/embedding_config_int8.yaml)
    - [reranker_config_fp16.yaml](./build_in/build/reranker_config_fp16.yaml)
    - [reranker_config_int8.yaml](./build_in/build/reranker_config_int8.yaml)

    ```bash
    vamc compile ./build_in/build/embedding_config_fp16.yaml
    ```

### step.4 模型推理
1. 参考瀚博训推软件生态链文档，获取模型推理工具：[vaststreamx v2.8+](../../docs/vastai_software.md)
2. runstream推理：[demo.py](./build_in/vsx/demo.py)
    - 配置模型路径等参数，推理脚本内指定的文本对

    ```bash
    python ./build_in/vsx/demo.py \
        --vacc_weight bge-m3-512-fp16/mod \
        --torch_weight bge/bge-m3 \
        --task embedding \
        --eval_engine vacc \
        --eval_dataset mteb-sts12-sts_test.jsonl \
        --seqlen 512
    ```

### step.5 性能精度
1. 参考瀚博训推软件生态链文档，获取模型性能测试工具：[vamp v2.4+](../../docs/vastai_software.md)
2. 性能测试
    - 配置vdsp参数：[embedding-vdsp_params.json](./build_in/vdsp_params/embedding-vdsp_params.json)

    ```bash
    vamp -m vacc_deploy/bge-m3-512-fp16/mod \
    --vdsp_params text2vec/common/vacc_code/vdsp_params/embedding-vdsp_params.json \
    -i 1 p 1 -b 1 -s [[1,512],[1,512],[1,512],[1,512],[1,512],[1,512]] --dtype uint32
    ```

- 精度测试：[demo.py](./build_in/vsx/demo.py)
    - 配置模型路径等参数，指定`--eval_mode`参数为True，进行精度评估

    ```bash
    python ./build_in/vsx/demo.py \
        --vacc_weight bge-m3-512-fp16/mod \
        --torch_weight bge/bge-m3 \
        --task embedding \
        --eval_mode \
        --eval_engine vacc \
        --eval_dataset mteb-sts12-sts_test.jsonl \
        --seqlen 512
    ```


### Tips
- reranker模型，不需要指定`output_layout`编译参数
- 注意模型本身只需3个输入，但编译器需要6个输入
- bge-m3模型，int8量化精度掉点验证，可使用以下量化参数实现混合精度量化，跳过一些层保留fp16
    ```yaml
    quantize:
        calibrate_mode: percentile
        quantize_per_channel: false
        overflow_adaptive: 1
        weight_scale: max
        calibrate_chunk_by: -1
        exclude_layers: [1, 2, 5, 6, 10, 11, 12, 15, 16, 19, 20, 22, 26, 27, 28, 31, 32, 35, 36, 38, 42, 43, 44, 47, 48, 51, 52, 54, 58, 59, 60, 63, 64, 67, 68, 70, 74, 75, 76, 79, 80, 83, 84, 86, 90, 91, 92, 95, 96, 99, 100, 102, 106, 107, 108, 111, 112, 115, 116, 118, 122, 123, 124, 127, 128, 131, 132, 134, 138, 139, 140, 143, 144, 147, 148, 150, 154, 155, 156, 159, 160, 163, 164, 166, 170, 171, 172, 175, 176, 179, 180, 182, 186, 187, 188, 191, 192, 195, 196, 198, 202, 203, 204, 207, 208, 211, 212, 214, 218, 219, 220, 223, 224, 227, 228, 230, 234, 235, 236, 239, 240, 243, 244, 246, 250, 251, 252, 255, 256, 259, 260, 262, 266, 267, 268, 271, 272, 275, 276, 278, 282, 283, 284, 287, 288, 291, 292, 294, 298, 299, 300, 303, 304, 307, 308, 310, 314, 315, 316, 319, 320, 323, 324, 326, 330, 331, 332, 335, 336, 339, 340, 342, 346, 347, 348, 351, 352, 355, 356, 358, 362, 363, 364, 367, 368, 371, 372, 374, 378, 379, 380, 384]
        quantize_operators: ['!_add']
    ```
- bge-reranker-v2-m3模型，int8量化精度掉点验证，可使用以下量化参数实现混合精度量化，跳过一些层保留fp16
    ```yaml
    quantize:
        calibrate_mode: max
        quantize_per_channel: false
        overflow_adaptive: 1
        weight_scale: max
        calibrate_chunk_by: -1
        exclude_layers: [1, 2, 5, 6, 10, 11, 12, 15, 16, 19, 20, 22, 26, 27, 28, 31, 32, 35, 36, 38, 42, 43, 44, 47, 48, 51, 52, 54, 58, 59, 60, 63, 64, 67, 68, 70, 74, 75, 76, 79, 80, 83, 84, 86, 90, 91, 92, 95, 96, 99, 100, 102, 106, 107, 108, 111, 112, 115, 116, 118, 122, 123, 124, 127, 128, 131, 132, 134, 138, 139, 140, 143, 144, 147, 148, 150, 154, 155, 156, 159, 160, 163, 164, 166, 170, 171, 172, 175, 176, 179, 180, 182, 186, 187, 188, 191, 192, 195, 196, 198, 202, 203, 204, 207, 208, 211, 212, 214, 218, 219, 220, 223, 224, 227, 228, 230, 234, 235, 236, 239, 240, 243, 244, 246, 250, 251, 252, 255, 256, 259, 260, 262, 266, 267, 268, 271, 272, 275, 276, 278, 282, 283, 284, 287, 288, 291, 292, 294, 298, 299, 300, 303, 304, 307, 308, 310, 314, 315, 316, 319, 320, 323, 324, 326, 330, 331, 332, 335, 336, 339, 340, 342, 346, 347, 348, 351, 352, 355, 356, 358, 362, 363, 364, 367, 368, 371, 372, 374, 378, 379, 380, 384]
        quantize_operators: ['!_add']
    ```