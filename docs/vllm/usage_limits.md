# Model Usage Limits in vLLM Backend

## NLP

### BGE

#### Supported Models
- Embedding
  | Model  | Dimension | Sequence Length | Language |
  | :------ | :------ | :------ | :------ |
  |      [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)       |   1024    |      8192       | multilingual |
  | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |    384    |       512       |  English   |
  | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |    768    |       512       |   English  |
  | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |   1024    |       512       |  English  |
  | [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) |    384    |       512       |  Chinese |
  | [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) |    768    |       512       |   Chinese |
  | [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) |   1024    |       512       |   Chinese |
    

- ReRanker

  | Model |   Base Model  |  Dimension | Sequence Length  |Language |
  | :------ | :------ | :------ | :------| :------ |
  | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | [bge-m3](https://huggingface.co/BAAI/bge-m3)  |  1024 | 8192 |  multilingual | 
  | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)  | 768 | 512 | English/Chinese  |  
  | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | [xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) |  1024 | 512 | English/Chinese  |

#### Usage Limits

- Embedding

  | model | parallel | seq limit | tips|
  | :------ | :------ | :------ | :------ |
  | bge-m3 | TP1/2/4/8 | max-model-len 8192 | max-concurrency 4|
  | bge-small-* | TP1/2/4/8 | max-model-len 512 | max-concurrency 4|
  | bge-base-* | TP1/2/4/8 | max-model-len 512 | max-concurrency 4|
  | bge-large-* | TP1/2/4/8 | max-model-len 512 | max-concurrency 4|

- ReRanker

  | model | parallel | seq limit | tips|
  |:------|:------ | :------ | :------ | 
  | bge-reranker-v2-m3 | TP1/2/4/8 | max-model-len 8192 | max-concurrency 4|
  | bge-reranker-base | TP1/2/4/8 | max-model-len 512 | max-concurrency 4|
  | bge-reranker-large | TP1/2/4/8 | max-model-len 512 | max-concurrency 4|


### Qwen3-Embedding/Reranker

#### Supported Models

- Embedding

  | Model Type | Models | Size | Layers | Sequence Length | Embedding Dimension | MRL Support | Instruction Aware |
  |:------|:------| :------| :------| :------| :------| :------|  :------| 
  | Text Embedding   | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 0.6B | 28     | 32K             | 1024                | Yes         | Yes            |
  | Text Embedding   | [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | 4B | 36     | 32K             | 2560                | Yes         | Yes            |
  | Text Embedding   | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 8B | 36     | 32K             | 4096                | Yes         | Yes            |
    

- ReRanker

  | Model Type   | Models | Size | Layers | Sequence Length | Embedding Dimension | MRL Support | Instruction Aware |
  |:------|:------| :------| :------| :------| :------| :------|  :------| 
  | Text Reranking   | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 0.6B | 28     | 32K             | -                   | -           | Yes            |
  | Text Reranking   | [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | 4B | 36     | 32K             | -                   | -           | Yes            |
  | Text Reranking   | [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | 8B | 36     | 32K             | -                   | -           | Yes            |


#### Usage Limits

- Embedding

  | model | parallel | seq limit | tips|
  | :------| :------| :------|  :------| 
  | Qwen3-Embedding-0.6B | TP1/2/4/8 | max-model-len 32k | max-concurrency 4|
  | Qwen3-Embedding-4B | TP1/2/4/8 | max-model-len 32k | max-concurrency 4|
  | Qwen3-Embedding-8B | TP2/4/8 | max-model-len 32k | max-concurrency 4|

- ReRanker

  | model | parallel | seq limit | tips|
  | :------| :------| :------|  :------| 
  | Qwen3-Reranker-0.6B | TP1/2/4/8 | max-model-len 32k | max-concurrency 4|
  | Qwen3-Reranker-4B | TP1/2/4/8 | max-model-len 32k |max-concurrency 4|
  | Qwen3-Reranker-8B | TP2/4/8 | max-model-len 32k | max-concurrency 4|


## LLM


### DeepSeek-V3

#### Supported Models

  |model | huggingface  | modelscope | parameter | dtype| parallel |
  | :--- | :--- | :-- | :-- | :-- | :-- |
  |DeepSeek-V3| [deepseek-ai/DeepSeek-V3](https://hf-mirror.com/deepseek-ai/DeepSeek-V3) | [deepseek-ai/DeepSeek-V3](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3) | 671B-A37B | FP8 | TP32 |
  |DeepSeek-V3-Base| [deepseek-ai/DeepSeek-V3-Base](https://hf-mirror.com/deepseek-ai/DeepSeek-V3-Base) | [deepseek-ai/DeepSeek-V3-Base](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3-Base) | 671B-A37B | FP8 |TP32 |
  |DeepSeek-V3-0324| [deepseek-ai/DeepSeek-V3-0324](https://hf-mirror.com/deepseek-ai/DeepSeek-V3-0324) | [deepseek-ai/DeepSeek-V3-0324](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3-0324) | 671B-A37B | FP8 |TP32 |
  |DeepSeek-V3.1| [deepseek-ai/DeepSeek-V3.1](https://hf-mirror.com/deepseek-ai/DeepSeek-V3.1) | [deepseek-ai/DeepSeek-V3.1](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3.1) | 671B-A37B | FP8 |TP32 |
  |DeepSeek-V3.1-Base| [deepseek-ai/DeepSeek-V3.1-Base](https://hf-mirror.com/deepseek-ai/DeepSeek-V3.1-Base) | [deepseek-ai/DeepSeek-V3.1-Base](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3.1-Base) | 671B-A37B | FP8 |TP32 |
  |DeepSeek-V3.1-Terminus| [deepseek-ai/DeepSeek-V3.1-Terminus](https://hf-mirror.com/deepseek-ai/DeepSeek-V3.1-Terminus) | [deepseek-ai/DeepSeek-V3.1-Terminus](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3.1-Terminus) | 671B-A37B | FP8 |TP32 |


#### Usage Limits

  | parallel | seq limit | mtp | tips|
  |:--- | :-- | :-- | :-- |
  | TP32 | max-input-len 56k </br> max-model-len 64k | ✅ | max-concurrency 4|
  | TP32-PP2 | max-input-len 100k </br> max-model-len 128k | ❌ | max-concurrency 4|


### DeepSeek-R1

#### Supported Models

  |model | huggingface  | modelscope | parameter | dtype| parallel |
  | :--- | :--- | :-- | :-- | :-- | :-- |
  |DeepSeek-R1| [deepseek-ai/DeepSeek-R1](https://hf-mirror.com/deepseek-ai/DeepSeek-R1) | [deepseek-ai/DeepSeek-R1](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1) | 671B-A37B | FP8 |TP32 |
  |DeepSeek-R1-0528| [deepseek-ai/DeepSeek-R1-0528](https://hf-mirror.com/deepseek-ai/DeepSeek-R1-0528) | [deepseek-ai/DeepSeek-R1](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-0528) | 671B-A37B | FP8 | TP32 |

#### Usage Limits

  | parallel | seq limit | mtp | tips|
  |:--- | :-- | :-- | :-- |
  | TP32 | max-input-len 56k </br> max-model-len 64k | ✅ | max-concurrency 4|
  | TP32-PP2 | max-input-len 100k </br> max-model-len 128k | ❌ | max-concurrency 4|


### Qwen2.5

#### Supported Models

  |model | huggingface  | modelscope | parameter | dtype| parallel |
  | :--- | :--- | :-- | :-- | :-- | :-- |
  | Qwen2.5-7B-Instruct-GPTQ-Int4|[Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4](https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4/) | [Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4) | 7B | INT4 | TP2/4 |
  | Qwen2.5-14B-Instruct-GPTQ-Int4|[Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4](https://hf-mirror.com/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4/) | [Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4](https://www.modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4) | 14B | INT4 | TP2/4 |
  | Qwen2.5-32B-Instruct-GPTQ-Int4|[Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4](https://hf-mirror.com/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4/) | [Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4](https://www.modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4) | 32B | INT4 | TP2/4 |


#### Usage Limits
  | model | parallel | seq limit | mtp | tips|
  |:--- |:--- | :-- | :-- | :-- |
  | Qwen2.5-7B-GPTQ-Int4 | TP2/4 | max-input-len 56k </br> max-model-len 64k | ❌ | max-concurrency 4|
  | Qwen2.5-14B-GPTQ-Int4 | TP2/4 | max-input-len 56k </br> max-model-len 64k | ❌ | max-concurrency 4|
  | Qwen2.5-32B-GPTQ-Int4 | TP4/8 | max-input-len 56k </br> max-model-len 64k | ❌ | max-concurrency 4|


### Qwen3

#### Supported Models
- LLM-Dense-GQA

  |model | huggingface  | modelscope | parameter | dtype| parallel |
  | :--- | :--- | :-- | :-- | :-- | :-- |
  | Qwen3-0.6B | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | [Qwen/Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B) | 0.6B | BF16 | TP2/4/8 |
  | Qwen3-0.6B-FP8 | [Qwen/Qwen3-0.6B-FP8](https://huggingface.co/Qwen/Qwen3-0.6B-FP8) | [Qwen/Qwen3-0.6B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B-FP8) | 0.6B | FP8 | TP2/4/8 |
  | Qwen3-0_6B_int4_awq | [AngelSlim/Qwen3-0_6B_int4_awq](https://huggingface.co/AngelSlim/Qwen3-0_6B_int4_awq) | [AngelSlim/Qwen3-0_6B_int4_awq](https://www.modelscope.cn/AngelSlim/Qwen3-0_6B_int4_awq) | 0.6B | INT4 | TP2/4/8 |
  | Qwen3-0.6B-GPTQ-Int4 | [JunHowie/Qwen3-0.6B-GPTQ-Int4](https://huggingface.co/JunHowie/Qwen3-0.6B-GPTQ-Int4) | [JunHowie/Qwen3-0.6B-GPTQ-Int4](https://www.modelscope.cn/JunHowie/Qwen3-0.6B-GPTQ-Int4) | 0.6B | INT4 | TP2/4/8 |
  | Qwen3-1.7B | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | [Qwen/Qwen3-1.7B](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B) | 1.7B | BF16 | TP2/4/8 |
  | Qwen3-1.7B-FP8 | [Qwen/Qwen3-1.7B-FP8](https://huggingface.co/Qwen/Qwen3-1.7B-FP8) | [Qwen/Qwen3-1.7B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B-FP8) | 1.7B | FP8 | TP2/4/8 |
  | Qwen3-1_7B_int4_awq | [AngelSlim/Qwen3-1_7B_int4_awq](https://huggingface.co/AngelSlim/Qwen3-1_7B_int4_awq) | [AngelSlim/Qwen3-1_7B_int4_awq](https://www.modelscope.cn/AngelSlim/Qwen3-1_7B_int4_awq) | 1.7B | INT4 | TP2/4/8 |
  | Qwen3-1.7B-GPTQ-Int4 | [JunHowie/Qwen3-1.7B-GPTQ-Int4](https://huggingface.co/JunHowie/Qwen3-1.7B-GPTQ-Int4) | [JunHowie/Qwen3-1.7B-GPTQ-Int4](https://www.modelscope.cn/JunHowie/Qwen3-1.7B-GPTQ-Int4) | 1.7B | INT4 | TP2/4/8 |
  | Qwen3-4B | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | [Qwen/Qwen3-4B](https://www.modelscope.cn/models/Qwen/Qwen3-4B) | 4B | BF16 | TP2/8 |
  | Qwen3-4B-FP8 | [Qwen/Qwen3-4B-FP8](https://huggingface.co/Qwen/Qwen3-4B-FP8) | [Qwen/Qwen3-4B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-4B-FP8) | 4B | FP8 | TP2/8 |
  | Qwen3-4B-Instruct-2507-FP8 | [Qwen/Qwen3-4B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8) | [Qwen/Qwen3-4B-Instruct-2507-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507-FP8) | 4B | FP8 | TP2/8 |
  | Qwen3-4B-Thinking-2507-FP8 | [Qwen/Qwen3-4B-Thinking-2507-FP8](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507-FP8) | [Qwen/Qwen3-4B-Thinking-2507-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-4B-Thinking-2507-FP8) | 4B | FP8 | TP2/8 |
  | Qwen3-4B-AWQ | [Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ) | [Qwen/Qwen3-4B-AWQ](https://www.modelscope.cn/models/Qwen/Qwen3-4B-AWQ) | 4B | INT4 | TP2/4* |
  | Qwen3-4B-GPTQ-Int4 | [JunHowie/Qwen3-4B-GPTQ-Int4](https://huggingface.co/JunHowie/Qwen3-4B-GPTQ-Int4) | [JunHowie/Qwen3-4B-GPTQ-Int4](https://www.modelscope.cn/models/JunHowie/Qwen3-4B-GPTQ-Int4) | 4B | INT4 | TP2/8 |
  | Qwen3-8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | [Qwen/Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B) | 8B | BF16 | TP2/8 |
  | Qwen3-8B-FP8 | [Qwen/Qwen3-8B-FP8](https://huggingface.co/Qwen/Qwen3-8B-FP8) | [Qwen/Qwen3-8B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-8B-FP8) | 8B | FP8 | TP2/8 |
  | Qwen3-8B-AWQ | [Qwen/Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ) | [Qwen/Qwen3-8B-AWQ](https://www.modelscope.cn/models/Qwen/Qwen3-8B-AWQ) | 8B | INT4 | TP2/4/8 |
  | Qwen3-8B-GPTQ-Int4 | [JunHowie/Qwen3-8B-GPTQ-Int4](https://huggingface.co/JunHowie/Qwen3-8B-GPTQ-Int4) | [JunHowie/Qwen3-8B-GPTQ-Int4](https://www.modelscope.cn/models/JunHowie/Qwen3-8B-GPTQ-Int4) | 8B | INT4 | TP2/8 |
  | Qwen3-14B | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | [Qwen/Qwen3-14B](https://www.modelscope.cn/models/Qwen/Qwen3-14B) | 14B | BF16 | TP2/4/8 |
  | Qwen3-14B-FP8 | [Qwen/Qwen3-14B-FP8](https://huggingface.co/Qwen/Qwen3-14B-FP8) | [Qwen/Qwen3-14B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-14B-FP8) | 14B | FP8 | TP2/4/8 |
  | Qwen3-14B-AWQ | [JunHowie/Qwen3-14B-AWQ](https://huggingface.co/JunHowie/Qwen3-14B-AWQ) | [JunHowie/Qwen3-14B-AWQ](https://www.modelscope.cn/models/JunHowie/Qwen3-14B-AWQ) | 14B | INT4 | TP2/4/8 |
  | Qwen3-14B-GPTQ-Int4 | [JunHowie/Qwen3-14B-GPTQ-Int4](https://huggingface.co/JunHowie/Qwen3-14B-GPTQ-Int4) | [JunHowie/Qwen3-14B-GPTQ-Int4](https://www.modelscope.cn/models/JunHowie/Qwen3-14B-GPTQ-Int4) | 14B | INT4 | TP2/4/8 |
  | Qwen3-14B-int4-AutoRound-gptq-inc | - | [Intel/Qwen3-14B-int4-AutoRound-gptq-inc](https://www.modelscope.cn/models/Intel/Qwen3-14B-int4-AutoRound-gptq-inc) | 14B | INT4 | TP2/4/8 |
  | Qwen3-32B | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | [Qwen/Qwen3-32B](https://www.modelscope.cn/models/Qwen/Qwen3-32B) | 32B | BF16 | TP4/8 |
  | Qwen3-32B-FP8 | [Qwen/Qwen3-32B-FP8](https://huggingface.co/Qwen/Qwen3-32B-FP8) | [Qwen/Qwen3-32B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-32B-FP8) | 32B | FP8 | TP4/8 |
  | Qwen3-32B-AWQ | [Qwen/Qwen3-32B-AWQ](https://huggingface.co/Qwen/Qwen3-32B-AWQ) | [Qwen/Qwen3-32B-AWQ](https://www.modelscope.cn/models/Qwen/Qwen3-32B-AWQ) | 32B | INT4 | TP4/8 |
  | Qwen3-32B-GPTQ-Int4 | [JunHowie/Qwen3-32B-GPTQ-Int4](https://huggingface.co/JunHowie/Qwen3-32B-GPTQ-Int4) | [JunHowie/Qwen3-32B-GPTQ-Int4](https://www.modelscope.cn/models/JunHowie/Qwen3-32B-GPTQ-Int4) | 32B | INT4 | TP4/8 |


- LLM-MOE-GQA

  |model | huggingface  | modelscope | parameter | dtype| parallel |
  | :--- | :--- | :-- | :-- | :-- | :-- |
  |Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4 | [lancew/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4](https://huggingface.co/lancew/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4) | / | 30B-A3B | INT4 | TP2/4 |
  |Qwen3-30B-A3B-Thinking-2507-GPTQ-Int4 | [lancew/Qwen3-30B-A3B-Thinking-2507-GPTQ-Int4](https://huggingface.co/lancew/Qwen3-30B-A3B-Thinking-2507-GPTQ-Int4) | / | 30B-A3B | INT4 | TP2/4 |
  |Qwen3-30B-A3B-GPTQ-Int4 | [Qwen/Qwen3-30B-A3B-GPTQ-Int4](https://hf-mirror.com/Qwen/Qwen3-30B-A3B-GPTQ-Int4) | [Qwen/Qwen3-30B-A3B-GPTQ-Int4](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B-GPTQ-Int4) | 30B-A3B | INT4 | TP2/4 |
  |Qwen3-30B-A3B-FP8 | [Qwen/Qwen3-30B-A3B-FP8](https://hf-mirror.com/Qwen/Qwen3-30B-A3B-FP8) | [Qwen/Qwen3-30B-A3B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B-FP8) | 30B-A3B | FP8 | TP2/4 |
  |Qwen3-30B-A3B-Instruct-2507-FP8 | [Qwen/Qwen3-30B-A3B-Instruct-2507-FP8](https://hf-mirror.com/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8) | [Qwen/Qwen3-30B-A3B-Instruct-2507-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8) | 30B-A3B | FP8 | TP2/4 |
  |Qwen3-30B-A3B-Thinking-2507-FP8 | [Qwen/Qwen3-30B-A3B-Thinking-2507-FP8](https://hf-mirror.com/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8) | [Qwen/Qwen3-30B-A3B-Thinking-2507-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8) | 30B-A3B | FP8 | TP2/4 |
  |Qwen3-Coder-30B-A3B-Instruct-FP8 | [Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8](https://hf-mirror.com/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8) | [Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8) | 30B-A3B | FP8 | TP2/4 |
  |Qwen3-235B-A22B-FP8 | [Qwen/Qwen3-235B-A22B-FP8](https://hf-mirror.com/Qwen/Qwen3-235B-A22B-FP8) | [Qwen/Qwen3-235B-A22B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B-FP8) | 235B-A22B | FP8 | TP16/32 |
  |Qwen3-235B-A22B-Instruct-2507-FP8 | [Qwen/Qwen3-235B-A22B-Instruct-2507-FP8](https://hf-mirror.com/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8) | [Qwen/Qwen3-235B-A22B-Instruct-2507-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8) | 235B-A22B | FP8 | TP16/32 |
  |Qwen3-235B-A22B-Thinking-2507-FP8 | [Qwen/Qwen3-235B-A22B-Thinking-2507-FP8](https://hf-mirror.com/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8) | [Qwen/Qwen3-235B-A22B-Thinking-2507-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8) | 235B-A22B | FP8 | TP16/32 |


#### Usage Limits

- LLM-Dense-GQA

  | model | parallel | seq limit | mtp | tips|
  |:--- |:--- | :-- | :-- | :-- |
  | Qwen3-0.6B-* | TP2/4/8 | max-model-len 32k | ❌ | max-concurrency 4|
  | Qwen3-1.7B-* | TP2/4/8 | max-model-len 32k | ❌ | max-concurrency 4|
  | Qwen3-4B-* | TP2 | max-input-len 56k </br> max-model-len 64k | ❌ | max-concurrency 4|
  | Qwen3-4B-* | TP4/8 | max-input-len 100k </br> max-model-len 128k | ❌ | max-concurrency 4|
  | Qwen3-8B-* | TP2 | max-input-len 56k </br> max-model-len 64k | ❌ | max-concurrency 4|
  | Qwen3-8B-* | TP4/8 | max-input-len 100k </br> max-model-len 128k | ❌ | max-concurrency 4|
  | Qwen3-14B-* | TP2/4 | max-input-len 56k </br> max-model-len 64k | ❌ | max-concurrency 4|
  | Qwen3-14B-* | TP8 | max-input-len 100k </br> max-model-len 128k | ❌ | max-concurrency 4|
  | Qwen3-32B-* | TP4 | max-input-len 56k </br> max-model-len 64k | ❌ | max-concurrency 4|
  | Qwen3-32B-* | TP8 | max-input-len 100k </br> max-model-len 128k | ❌ | max-concurrency 4|

- LLM-MOE-GQA

  | model | parallel | seq limit | mtp | tips|
  |:--- |:--- | :-- | :-- | :-- |
  | Qwen3-30B-A3B-* | TP2 | max-input-len 56k </br> max-model-len 64k | ❌ | max-concurrency 4|
  | Qwen3-30B-A3B-* | TP4 | max-input-len 100k </br> max-model-len 128k | ❌ | max-concurrency 4|
  | Qwen3-235B-A22B-* | TP16 | max-input-len 100k </br> max-model-len 128k | ❌ | max-concurrency 4|
  | Qwen3-235B-A22B-* | TP32 | max-input-len 100k </br> max-model-len 128k | ❌ | max-concurrency 4|


### Tongyi-DeepResearch


#### Supported Models

  |model | huggingface  | modelscope | parameter | dtype| parallel |
  | :--- | :--- | :-- | :-- | :-- | :-- |
  |Tongyi-DeepResearch-30B-A3B-FP8 | [lancew/Tongyi-DeepResearch-30B-A3B-FP8](https://huggingface.co/lancew/Tongyi-DeepResearch-30B-A3B-FP8) | - | 30B-A3B | FP8 |TP2/4 |


#### Usage Limits

  | model | parallel | seq limit | mtp | tips|
  |:--- |:--- | :-- | :-- | :-- |
  | Tongyi-DeepResearch-30B-A3B-FP8 | TP2 | max-input-len 56k </br> max-model-len 64k | ❌ | max-concurrency 4|
  | Tongyi-DeepResearch-30B-A3B-FP8 | TP4 | max-input-len 100k </br> max-model-len 128k | ❌ | max-concurrency 4|


### MiroThinker-v1.5

#### Supported Models

  | model   | huggingface  | base_model  | parameter | dtype | parallel   |
  | :---- | :---- | :---- | :---- | :---- |:---- |
  | MiroThinker-v1.5-30B-FP8  | [lancew/MiroThinker-v1.5-30B-FP8](https://huggingface.co/lancew/MiroThinker-v1.5-30B-FP8)   | Qwen3-30B-A3B-Thinking-2507   | 30B-A3B |FP8   | TP2/4 |
  | MiroThinker-v1.5-235B-FP8 | [lancew/MiroThinker-v1.5-235B-FP8](https://huggingface.co/lancew/MiroThinker-v1.5-235B-FP8) | Qwen3-235B-A22B-Thinking-2507 | 235B-A22B | FP8   | TP16/32 |


#### Usage Limits

  | model   | parallel | seq limit | mtp  | tips  |
  | :---- | :---- | :---- | :---- | :---- |
  | MiroThinker-v1.5-30B-FP8  | TP2      | max-input-len 56k </br> max-model-len 64k   | ❌    | max-concurrency 4 |
  | MiroThinker-v1.5-30B-FP8  | TP4      | max-input-len 100k </br> max-model-len 128k | ❌    | max-concurrency 4 |
  | MiroThinker-v1.5-235B-FP8 | TP16/TP32 | max-input-len 100k </br> max-model-len 128k | ❌    | max-concurrency 4 |


## VLM

### Qwen3-VL

#### Supported Models

  |model | huggingface  | modelscope | parameter | dtype| parallel |
  | :--- | :--- | :-- | :-- | :-- | :-- |
  |Qwen3-VL-30B-A3B-Instruct-FP8 | [Qwen/Qwen3-VL-30B-A3B-Instruct-FP8](https://hf-mirror.com/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8) | [Qwen/Qwen3-VL-30B-A3B-Instruct-FP8](https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8) | 30B-A3B | FP8 |TP2/4 |
  |Qwen3-VL-30B-A3B-Thinking-FP8 | [Qwen/Qwen3-VL-30B-A3B-Thinking-FP8](https://hf-mirror.com/Qwen/Qwen3-VL-30B-A3B-Thinking-FP8) | [Qwen/Qwen3-VL-30B-A3B-Thinking-FP8](https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Thinking-FP8) | 30B-A3B | FP8 |TP2/4 |


#### Usage Limits

  | model | parallel | seq limit | mtp | tips|
  |:--- |:--- | :-- | :-- | :-- |
  | Qwen3-VL-30B-A3B-*-FP8 | tp2 | max-input-len 56k </br> max-model-len 64k | ❌ | max-concurrency 4|
  | Qwen3-VL-30B-A3B-*-FP8 | tp4 | max-input-len 100k </br> max-model-len 128k | ❌ | max-concurrency 4|

### MinerU

#### Supported Models

  | model | huggingface  | modelscope | parameter | dtype| parallel |
  | :--- | :--- | :-- | :-- | :-- | :-- |
  | MinerU2.5-2509-1.2B | [opendatalab/MinerU2.5-2509-1.2B](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) | [opendatalab/MinerU2.5-2509-1.2B](https://modelscope.cn/models/opendatalab/MinerU2.5-2509-1.2B) | 1.2B | BF16 | TP1/2 |


#### Usage Limits

  | model | parallel | seq limit | mtp | tips|
  |:--- |:--- | :-- | :-- | :-- |
  |  MinerU2.5-2509-1.2B | TP1/2 | max-model-len 16k | ❌ | max-concurrency 4|


## TIPS
- 对于超过上下文长度的请求，内部会拦截不做处理，需要用户端自行处理