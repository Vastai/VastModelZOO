# NLP
## 1. arch summary
![总览](../images/algorm_cate_arch/transformer.png)

|                                                                   title                                                                   |    name     |  codebase |   acrh    |   time |
| :---------------------------------------------------------------------------------------------------------------------------------------: | :---------: |:---------: |:---------:  | ---- |
|   [Attention Is All You Need](https://arxiv.org/abs/1706.03762)               |    Transformer     | [github](https://github.com/tensorflow/tensor2tensor) | Seq2Seq | 2017 |
|   [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)               |    GPT1     | [github](https://github.com/openai/finetune-transformer-lm) | Decoder | 2018 |
|   [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)               |    GPT2     | [github](https://github.com/openai/gpt-2) | Decoder | 2019 |
|   [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)               |    GPT3     | [github](https://github.com/openai/gpt-3) | Decoder | 2021 |
|   [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)               |    BERT     | [github](https://github.com/google-research/bert) | Encoder | 2018 |
|   [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)               |    RoBERTa     | [github](https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/roberta/model.py) | Encoder | 2019 |
|   [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)               |    XLM-RoBERTa     | [github](https://github.com/facebookresearch/XLM) | Encoder | 2019 |
|   [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)               |    ALBERT     | [github](https://github.com/google-research/albert) | Encoder | 2020 |
|   [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)               |    DistilBERT     | [github](https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py) | Encoder | 2020 |
|   [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)               |    Transformer-XL    | [github](https://github.com/kimiyoung/transformer-xl) | Decoder | 2019 |
|   [XLNet:Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf)               |    XLNet    | [github](https://github.com/zihangdai/xlnet) | Decoder | 2019 |
|   [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223v1)               |    ERNIE1.0    | [github](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-1.0) | Encoder | 2019 |
|   [ERNIE 2.0: A Continual Pre-training Framework for Language Understanding](https://arxiv.org/abs/1907.12412)               |    ERNIE2.0    | [github](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-1.0) | Encoder | 2019 |
|   [ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2107.02137)               |    ERNIE3.0    | [github](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0) | Encoder | 2021 |
|   [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf)               |     Longformer    | [github](https://github.com/allenai/longformer) | Encoder | 2020 |
|   [Big Bird: Transformers for Longer Sequences](https://arxiv.org/pdf/2007.14062.pdf)               |     BigBird    | [github](https://github.com/google-research/bigbird/blob/master/bigbird/core/attention.py) | Encoder | 2020 |
|   [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1909.00204)               |     BART    | [github](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nezha/modeling_nezha.py) | Seq2Seq | 2020 |
|   [NEZHA: Neural Contextualized Representation for Chinese Language Understanding](https://arxiv.org/abs/1909.00204)               |     NEZHA    | [github](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nezha/modeling_nezha.py) | Encoder | 2019 |
|   [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)               |     T5    | [github](https://github.com/google-research/text-to-text-transfer-transformer) | Seq2Seq | 2020 |
|   [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB)               |     ELECTRA    | [github](https://github.com/google-research/electra) | Encoder | 2020 |
|   [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)               |     DeBERTa    | [github](https://github.com/microsoft/DeBERTa) | Seq2Seq | 2020 |

## 2. common module

### 2.1 arch

- Encoder
- Decoder
- Seq2Seq

## 3. special module
- [LayerNorm](https://arxiv.org/abs/1607.06450)
- [GELU](https://arxiv.org/abs/1810.04805)
- [Recurence](https://arxiv.org/abs/1901.02860)

## 4. op list
- [model_op_details](./outfiles)
- [op list_summary](./outfiles/op_summary.md)
