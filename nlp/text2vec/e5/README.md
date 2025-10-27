# E5

> [huggingface](https://huggingface.co/intfloat/multilingual-e5-base)



## Embedding

> base XLMRobertaModel

|                          Model Name                          | Dimension | Sequence Length |                         Introduction                         |
| :----------------------------------------------------------: | :-------: | :-------------: | :---------------------------------------------------------- |
|      [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large/)       |   1024    |      512       | multilingual |
|      [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)       |   1024    |      512       | multilingual |
|      [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)       |   1024    |      512       | multilingual |
|      [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)       |   1024    |      512       | multilingual |


### model analysis
model arch为XLMRobertaModel, 模型分为四层，分别是前处理层、Embedding(XLMRobertaEmbeddings)、Encoder(XLMRobertaEncoder)以及Pooler(XLMRobertaPooler)层。
#### pre-process for XLMRobertaModel
1. query + corpus的输入形式
   - multilingual-e5-large-instruct
    
     模型接受的输入为sentence list，每个sentence可以为带有Instruct的query的形式：
     ```python 
      f'Instruct: {task_description}\nQuery: {query}'
     ```
     ，也可以单独的sentence语料。
   - multilingual-e5-large/base/small
    
     模型接受的输入为sentence list， query不带Instruct。
     根据计算出来的embedding输出，对queries和corpus进行相似都计算。
2. tokenizer
   - input_ids

     分词器会将输入的sentence batches中个每个sentence中的token，按照在词表中的索引存储在input_ids中。

   - attention_mask

     为了避免在padding token上执行attention，分词器在padding token索引上进行了mask，取值范围是0和1，0表示进行掩码，1表示不进行掩码。

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
      没有构造转换的见[BERT#position_ids](../ACGE/README.md)。
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


#### XLMRobertaModel Embedding(XLMRobertaEmbeddings)
1. Embedding结构

   XLMRobertaModel使用embedding为XLMRobertaEmbeddings
   ```python
    self.embeddings = XLMRobertaEmbeddings(config)
   ```
   ```yaml
   (embeddings): XLMRobertaEmbeddings(
    (word_embeddings): Embedding(250002, 1024, padding_idx=1)
    (position_embeddings): Embedding(514, 1024, padding_idx=1)
    (token_type_embeddings): Embedding(1, 1024)
    (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
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

#### XLMRobertaModel Encoder(XLMRobertaEncoder)

    利用Embeding层输出进行encode。
```yaml
(encoder): XLMRobertaEncoder(
    (layer): ModuleList(
      (0-23): 24 x XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=1024, out_features=1024, bias=True)
            (key): Linear(in_features=1024, out_features=1024, bias=True)
            (value): Linear(in_features=1024, out_features=1024, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=1024, out_features=1024, bias=True)
            (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=1024, out_features=4096, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=4096, out_features=1024, bias=True)
          (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
```
#### XLMRobertaModel Pooler(XLMRobertaPooler)
    目前我们配置的pooler为None。即得到的是不进行XLMRobertaPooler之后的输出。
    在模型加载的时候可以利用`add_pooling_layer= False/True`来配置是否增加Poller 层。
  ```yaml
    pooler): XLMRobertaPooler(
    (dense): Linear(in_features=1024, out_features=1024, bias=True)
    (activation): Tanh()
  )
  ```

#### 其他后处理Pooling + Normalize
- multilingual-e5 官网提供的average_pool+normalize
  
  针对multilingual-e5-base/smalle/large/large-instruct，提供了关于average_pool+Normalization的后处理, 根据每一条sentence的embedding，和对应sentence中token的数量进行pool。
  同样在sentencetransformer中，等同于pooling_mode_mean_tokens=True的配置。

  ```python

  def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
      last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
      return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
  ...
  # average_pool步骤等同于sentencetransformer中的pooling_mode_mean_tokens=True && pooling_mode_mean_sqrt_len_tokens=False的配置
  embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
  
  # normalize embeddings
  embeddings = F.normalize(embeddings, p=2, dim=1)

  ```
  

- sentencetransformer
    sentence_transformers~=2.2.2

    在模型加载的时候，可还以根据modules.json进行不同层的加载，本模型中除了上述的和关于XLMRobertaModel不同层即Transformer的加载之外，还配置了Pooling和Normalize，Normalize具体的配置没有给出。当使用SentenceTransformer进行加载的时候，可以利用Pooling 或者 Normalize的配置，对输出进行不同的后处理。
sentencetransformer Pooling中的后处理策略：

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

## Build_In Deploy

- 参考：[bge/README.md](../bge/README.md)