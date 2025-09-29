# BCE

- Technical Report
  - https://zhuanlan.zhihu.com/p/681370855
- Huggingface
  - https://huggingface.co/maidalun1020/bce-embedding-base_v1
  - https://huggingface.co/maidalun1020/bce-reranker-base_v1


## Embedding

model arch为XLMRobertaModel, 模型分为四层，分别是前处理层、Embedding(XLMRobertaEmbeddings)、Encoder(XLMRobertaEncoder)以及Pooler(XLMRobertaPooler)层。

### pre-process for XLMRobertaModel
1. query + corpus的输入形式
  
    模型接受的输入为sentence list，query不带Instruct。根据计算出来的embedding输出，对queries和corpus进行相似都计算。

2. tokenizer
   - input_ids

     分词器会将输入的sentence batches中个每个sentence中的token，按照在词表中的索引存储在input_ids中.

   - attention_mask

     为了避免在padding token上执行attention，分词器在padding token索引上进行了mask，取值范围是0和1，0表示进行掩码，1表示不进行掩码

   - (optional) token_type_ids

     用于在相同的输入序列中编码不同句子，取值范围是0和1，0表示第一个句子，1表示第二个句子。

     在transformer的XLMRobertaModel中，token_type_ids在构建XLMRobertaModel的embedding XLMRobertaEmbeddings的时候，自动注册了全0的token_type_ids,shape为[1, max_position_embeddings]：

      ```python
       self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
     ```

     在XLMRobertaModel前向推理的过程中，会根据XLMRobertaEmbeddings注册的token_type_ids以及 batcnh_size和seq_length的大小，对token_type_ids进行构建扩充：
      ```python
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

     ```
      在vsx的推理中，在构造输入的前处理的过程中，同样构造了全0的token_type_ids， shape同input_ids.shape:
     
     ```python
     np.zeros(features['input_ids'].shape, dtype=np.int32)
     ```

3. Embedding中涉及的输入
    - position_ids
      
      在transformer中，XLMRobertaEmbeddings初始化的过程中注册了position_ids：
      ```python
       self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
      ```
      在XLMRobertaEmbeddings forward的过程中对position_ids进行了构造：
      ```python
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
      ```

      需要注意的是，在XLMRobertaEmbeddings中，不同于BertEmbeddings，使用了input_ids和self.padding_idx对position_ids进行了构造转换。
      这种转换构造，是利用了input_ids表达出来的padding信息，在进行position_embeddings计算之后，依然保持padding值。
      即假设input_ids中元素值t表示padding，那么转换保证了在计算position_embeddings(t对用的position_id)的时候，计算出来的embedding依然为paddind对应的值，一般情况下为全0张量。
      根据假设举例：
      - 情况一：假设input_ids中有t，[[t-1, t, t+1, t+2, t+3, t+4], [t, t+1, t-1, t+2, t+3, t+4]]， 那么构造是从其他的最初不为t的位置用值为t+1开始构造position_ids，遇到t则保留[[t+1, t, t+2, t+3, t+4, t+5], [t, t+1, t+2, t+3, t+4, t+5]]
      - 情况二：假设input_ids中无t，[[t+3, t+4, t+5, t+6, t+7, t+8], [t+4, t+5, t+6, t+7, t+8, t-1]]， 那么构造是从其他最初不为t的位置用值为t+1开始构造position_ids，[[t+1, t+2, t+3, t+4, t+5, t+6],[t+1, t+2, t+3, t+4, t+5, t+6]]
      在经过position_embeddings算子计算的时候，position_embeddings(t)会映射到position_embeddings层weight全0的计算中。其他的位置position embedding的计算正常进行。

      所以在vsx的推理过程中，替换了forward的过程，在forward中定义了position_ids:
      ```python
        if position_ids is None:
            position_ids = torch.Tensor(
                [[i for i in range(input_ids.shape[1])] for j in range(input_ids.shape[0])]
            ).to(dtype=input_ids.dtype)
      ```
      同时也针对有padding_idx的position_embeddings层，针对position_embeddings.weight进行了处理。


### XLMRobertaModel Embedding(XLMRobertaEmbeddings)
1. Embedding结构

   XLMRobertaModel使用embedding为XLMRobertaEmbeddings
   ```python
    self.embeddings = XLMRobertaEmbeddings(config)
   ```
   ```yaml
   (embeddings): XLMRobertaEmbeddings(
    (word_embeddings): Embedding(250002, 768, padding_idx=1)
    (position_embeddings): Embedding(514, 768, padding_idx=1)
    (token_type_embeddings): Embedding(1, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
   )
   ```
   在transformer的XLMRobertaEmbeddings层中，主要有3个embedding算子，相对于BertEmbeddings，position_embeddings增加了padding_idx的选项：
   ```python
    self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
   ...
   self.padding_idx = config.pad_token_id
   self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
   ```
2. XLMRobertaEmbeddings forward过程

   forward利用前处理得到数据进行，调用关系如下：
   - word_embeddings

      word_embeddings的输入为前处理得到的input_ids：
      ```python
      if inputs_embeds is None:
         inputs_embeds = self.word_embeddings(input_ids)
      ```
   - token_type_embeddings
      token_type_embeddings的输入为前处理token_type_ids：
      ```python
       token_type_embeddings = self.token_type_embeddings(token_type_ids)
       embeddings = inputs_embeds + token_type_embeddings
      ```
   - position_embeddings

     position_embeddings的输入为forward中构造的position_ids:
     ```python
      if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
      ```
   - post-process
 
     根据上面三个步骤得到的embedding进行LayerNorm和dropout,得到最终的embedding
     ```python
     embeddings = self.LayerNorm(embeddings)
     embeddings = self.dropout(embeddings) 
     ```

### XLMRobertaModel Encoder(XLMRobertaEncoder)

  利用Embeding层输出进行encode。

  ```yaml
  (encoder): XLMRobertaEncoder(
      (layer): ModuleList(
        (0-11): 12 x XLMRobertaLayer(
          (attention): XLMRobertaAttention(
            (self): XLMRobertaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): XLMRobertaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): XLMRobertaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): XLMRobertaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  ```

### XLMRobertaModel Pooler(XLMRobertaPooler)
  目前我们配置的pooler为None。即得到的是不进行XLMRobertaPooler之后的输出。

  在模型加载的时候可以利用`add_pooling_layer= False/True`来配置是否增加Poller层。

  ```yaml
     (pooler): XLMRobertaPooler(
     (dense): Linear(in_features=768, out_features=768, bias=True)
     (activation): Tanh()
     )
  ```

### 其他后处理Pooling + Normalize

- BCEmbedding提供了 cls和mean的pooler， 和默认的normalization
  ```python
  if self.pooler == "cls":
    embeddings = outputs.last_hidden_state[:, 0]
  elif self.pooler == "mean":
    attention_mask = inputs_on_device['attention_mask']
    last_hidden = outputs.last_hidden_state
    embeddings = (last_hidden * attention_mask.unsqueeze(-1).float()).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

  # embeddings normalization
  if normalize_to_unit:
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
  ```

- sentencetransformer
    在模型加载的时候，可还以根据modules.json进行不同层的加载，本模型中除了上述的和关于XLMRobertaModel不同层即Transformer的加载之外，还配置了Pooling和Normalize，Normalize具体的配置没有给出。
    
    当使用SentenceTransformer进行加载的时候，可以利用Pooling 或者 Normalize的配置，对输出进行不同的后处理。sentencetransformer Pooling中的后处理策略：

  <details>
    <summary>展开查看</summary>
    <pre><code>   
    if self.pooling_mode_cls_token:
            cls_token = features.get("cls_token_embeddings", token_embeddings[:, 0])  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        if self.pooling_mode_weightedmean_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            # token_embeddings shape: bs, seq, hidden_dim
            weights = (
                torch.arange(start=1, end=token_embeddings.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .to(token_embeddings.dtype)
                .to(token_embeddings.device)
            )
            assert weights.shape == token_embeddings.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        if self.pooling_mode_lasttoken:
            bs, seq_len, hidden_dim = token_embeddings.shape
            # attention_mask shape: (bs, seq_len)
            # Get shape [bs] indices of the last token (i.e. the last token for each batch item)
            # Use flip and max() to get the last index of 1 in the attention mask

            if torch.jit.is_tracing():
                # Avoid tracing the argmax with int64 input that can not be handled by ONNX Runtime: https://github.com/microsoft/onnxruntime/issues/10068
                attention_mask = attention_mask.to(torch.int32)

            values, indices = attention_mask.flip(1).max(1)
            indices = torch.where(values == 0, seq_len - 1, indices)
            gather_indices = seq_len - indices - 1

            # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (bs, 1, hidden_dim)

            # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
            # Actually no need for the attention mask as we gather the last token where attn_mask = 1
            # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
            # use the attention mask to ignore them again
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            embedding = torch.gather(token_embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
            output_vectors.append(embedding)

        output_vector = torch.cat(output_vectors, 1)
        features["sentence_embedding"] = output_vector
    </code></pre>
  </details>


## Reranker

model arch为XLMRobertaForSequenceClassification, roberta(XLMRobertaModel)，以及classifier(XLMRobertaClassificationHead)层。

其中roberta(XLMRobertaModel)包含XLMRobertaEmbeddings和XLMRobertaEncoder。

1.  pre-process for XLMRobertaForSequenceClassification

    前处理同`Embedding` 模型，区别在于输入为 `List`  的文本对
    ```python
    query = 'input_query'
    passages = ['passage_0', 'passage_1', ...]
    # construct sentence pairs
    sentence_pairs = [[query, passage] for passage in passages]
    ```
2. roberta模型结构

    2.1 XLMRobertaEmbeddings
    
    同`Embedding` 模型的 `XLMRobertaModel Embedding(XLMRobertaEmbeddings)`

    2.2 XLMRobertaEncoder

    同`Embedding` 模型的 `XLMRobertaModel Encoder(XLMRobertaEncoder)`

3. classifier(XLMRobertaClassificationHead)
   
   增加了分类后处理，结果shape为`[bs, 1]`。
   ```yaml
   (classifier): XLMRobertaClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (out_proj): Linear(in_features=768, out_features=1, bias=True)
   )
   ```
  

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
  |      [maidalun1020/bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1)       |   768    |      512       | multilingual; No need for "instruction" |

  >  base on XLMRobertaModel

- ReRanker

  |                          Model Name                          | Dimension | Sequence Length |                         Introduction                         |
  | :----------------------------------------------------------: | :-------: | :-------------: | :----------------------------------------------------------: |
  |      [maidalun1020/bce-reranker-base_v1](https://huggingface.co/maidalun1020/bce-reranker-base_v1)       |   768    |      512       | multilingual; No need for "instruction" |
  
  > base on XLMRobertaForSequenceClassification


2. 模型导出onnx：[onnx_export.py](../common/source_code/onnx_export.py)

    ```bash
    python ../common/source_code/onnx_export.py \
        --model bce-embedding-base_v1 \
        --type embedding \
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
1. 根据具体模型修改模型转换配置文件
    - [embedding_config_fp16.yaml](./build_in/build/embedding_config_fp16.yaml)
    - [embedding_config_int8.yaml](./build_in/build/embedding_config_int8.yaml)
    - [reranker_config_fp16.yaml](./build_in/build/reranker_config_fp16.yaml)
    - [reranker_config_int8.yaml](./build_in/build/reranker_config_int8.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    vamc compile ./build_in/build/embedding_config_fp16.yaml
    ```

### step.4 模型推理
1. runstream推理：[demo.py](./build_in/vsx/demo.py)
    - 配置模型路径等参数，推理脚本内指定的文本对

    ```bash
    python ./build_in/vsx/demo.py \
        --vacc_weight bce-embedding-base_v1-512-fp16/mod \
        --torch_weight bge/bce-embedding-base_v1 \
        --task embedding \
        --eval_engine vacc \
        --eval_dataset mteb-sts12-sts_test.jsonl \
        --seqlen 512
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数：[embedding-vdsp_params.json](./build_in/vdsp_params/embedding-vdsp_params.json)

    ```bash
    vamp -m vacc_deploy/bce-embedding-base_v1-512-fp16/mod \
    --vdsp_params text2vec/common/vacc_code/vdsp_params/embedding-vdsp_params.json \
    -i 1 p 1 -b 1 -s [[1,512],[1,512],[1,512],[1,512],[1,512],[1,512]] --dtype uint32
    ```
2. 精度测试：[demo.py](./build_in/vsx/demo.py)
    - 配置模型路径等参数，指定`--eval_mode`参数为`True`，进行精度评估

    ```bash
    python ./build_in/vsx/demo.py \
        --vacc_weight bce-embedding-base_v1-512-fp16/mod \
        --torch_weight bge/bce-embedding-base_v1 \
        --task embedding \
        --eval_mode \
        --eval_engine vacc \
        --eval_dataset mteb-sts12-sts_test.jsonl \
        --seqlen 512
    ```


### Tips
- reranker模型，不需要指定`output_layout`编译参数