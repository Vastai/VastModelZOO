# m3e

> [huggingface](https://huggingface.co/moka-ai/m3e-base)


## Embedding

> base BertModel

|                          Model Name                          | Dimension | Sequence Length |                         Introduction                         |
| :----------------------------------------------------------: | :-------: | :-------------: | :----------------------------------------------------------: |
|      [moka-ai/m3e-small](https://huggingface.co/moka-ai/m3e-small)       |   512    |      512       | 中文|
|      [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base)       |   768    |      512       | 主要是中文，少量英文 |
|      [moka-ai/m3e-large](https://huggingface.co/moka-ai/m3e-large)       |   768    |      512       | 中文 |


### model analysis
model arch为BertModel, 模型分为四层，分别是前处理层、Embedding(BertEmbeddings)、Encoder(BertEncoder)以及Pooler(BertPooler)层。

#### pre-process for BertModel
1. query + corpus的输入形式
 
     模型接受的输入为sentence list，query不带Instruct。根据计算出来的embedding输出，对queries和corpus进行相似都计算。
2. tokenizer
   - input_ids

     分词器会将输入的sentence batches中个每个sentence中的token，按照在词表中的索引存储在input_ids中

   - attention_mask

     为了避免在padding token上执行attention，分词器在padding token索引上进行了mask，取值范围是0和1，0表示进行掩码，1表示不进行掩码

   - (optional) token_type_ids

     用于在相同的输入序列中编码不同句子，取值范围是0和1，0表示第一个句子，1表示第二个句子。

     在transformer的BertModel中，token_type_ids在构建BertModel的embedding BertEmbeddings的时候，自动注册了全0的token_type_ids,shape为[1, max_position_embeddings]：

      ```python
       self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
     ```

     在BertModel前向推理的过程中，会根据BertEmbeddings注册的token_type_ids以及 batcnh_size和seq_length的大小，对token_type_ids进行构建扩充：
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
      
      在transformer中，BertEmbeddings初始化的过程中注册了position_ids：
      ```python
       self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
      ```
      在BertEmbeddings forward的过程中对position_ids进行了构造：
      ```python
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
      ```
      所以在vsx的推理过程中，替换了forward的过程，在forward中定义了position_ids:
      ```python
        if position_ids is None:
            position_ids = torch.Tensor(
                [[i for i in range(input_ids.shape[1])] for j in range(input_ids.shape[0])]
            ).to(dtype=input_ids.dtype)
      ```


#### BertModel Embedding(BertEmbeddings)
1. Embedding结构

   BertModel使用embedding为BertEmbeddings
   ```python
    self.embeddings = BertEmbeddings(config)
   ```
   ```yaml
   (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(21128, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
   ```
   在transformer的BertEmbeddings层中，主要有3个embedding算子：
   ```python
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
   ```
2. BertEmbeddings forward过程

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

#### BertModel Encoder(BertEncoder)

    利用Embeding层输出进行encode。
      
  ```yaml
  (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  ```
#### BertModel Pooler(BertPooler)
    目前我们配置的pooler为None。即得到的是不进行BertPooler之后的输出。
    在模型加载的时候可以利用`add_pooling_layer= False/True`来配置是否增加Poller 层。

  ```yaml
    (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
    )
  ```

#### 其他后处理Pooling + Normalize

在模型加载的时候，可还以根据modules.json进行不同层的加载，本模型中除了上述的和关于BertModel不同层即Transformer的加载之外，还配置了Pooling和Normalize，Normalize具体的配置没有给出。当使用SentenceTransformer进行加载的时候，可以利用Pooling 或者 Normalize的配置，对输出进行不同的后处理。
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

## Fine-tune

- [uniem 微调](https://github.com/wangyuxinwhy/uniem/blob/main/examples/finetune.ipynb)

```python
from datasets import load_dataset

from uniem.finetuner import FineTuner

dataset = load_dataset('shibing624/nli_zh', 'STS-B')
# 指定训练的模型为 m3e-small
finetuner = FineTuner.from_pretrained('moka-ai/m3e-small', dataset=dataset)
finetuner.run(epochs=1)
```

## Build_In Deploy
- 参考：[bge/README.md](../bge/README.md)