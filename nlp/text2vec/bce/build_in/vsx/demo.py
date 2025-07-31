# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import logging
import argparse
import numpy as np
from typing import Dict, List, Union

import torch
import datasets
import torch.nn.functional as F
from tqdm import tqdm
from tqdm.autonotebook import trange
from transformers import AutoTokenizer

import vaststreamx as vsx
from base import EmbeddingX


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%m/%d/%Y-%H:%M:%S",
    handlers=[logging.NullHandler(), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def cosine_similarity(A, B):
    return np.dot(A.flatten(), B.flatten()) / (np.linalg.norm(A) * np.linalg.norm(B))


class Embedding:
    def __init__(
        self,
        vacc_model_prefix_path: Union[str, Dict[str, str]],
        torch_model_or_tokenizer: str,
        onnx_model_path: str,
        device_id: int = 0,
        batch_size: int = 4,
        max_seqlen: int = 512,
    ):

        self.max_seqlen = max_seqlen
        self.torch_model_or_tokenizer = torch_model_or_tokenizer
        self.onnx_model_path = onnx_model_path
        self.batch_size = batch_size
        self.device_id = device_id
        self.vacc_model_prefix_path = vacc_model_prefix_path

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.torch_model_or_tokenizer, trust_remote_code=True
        )

    def init_vsx(self):
        # init vsx
        vsx_engine = EmbeddingX(
            model_prefix_path=self.vacc_model_prefix_path,
            device_id=self.device_id,
            batch_size=self.batch_size,
        )
        logger.info("vsx model init done.")
        return vsx_engine

    def init_torch(self):
        from transformers import AutoModel

        torch_engine = (
            AutoModel.from_pretrained(self.torch_model_or_tokenizer)
            .eval()
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
        # print("june model", torch_engine)
        return torch_engine

    def init_onnx(self):
        import onnxruntime
        onnx_engine = onnxruntime.InferenceSession(self.onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        return onnx_engine

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def infer(self, engine, sentences: Union[str, List[str]]) -> List:

        all_embeddings = []
        all_mask = []

        if isinstance(sentences, str):
            sentences = [sentences]
            
        # length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        # sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        disable = False
        if len(sentences) < 3:
            # do not show process bar
            disable = True

        for start_index in trange(0, len(sentences), self.batch_size, desc="Batches", disable=disable):
            sentences_batch = sentences[start_index : start_index + self.batch_size]
            features = self.tokenizer(
                sentences_batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_seqlen,
                return_tensors="np",
            )
            if isinstance(engine, torch.nn.Module):
                token_embeddings = self.run_torch(engine, features)
            elif isinstance(engine, EmbeddingX):
                token_embeddings = self.run_vsx(engine, features)
            else:
                token_embeddings = self.run_onnx(engine, features)

            token_embeddings = self.post_precess(token_embeddings)

            all_embeddings.extend(token_embeddings)
        
        return np.array(all_embeddings)

    def run_vsx(self, vsx_engine, features: Dict[str, np.ndarray]):
        # add for vsx
        features['token_type_ids'] = np.zeros(features['input_ids'].shape, dtype=np.int32)
        # default order array
        vsx_inputs = [
            features['input_ids'],
            features['attention_mask'],
            features['token_type_ids'],
            features['attention_mask'],
            features['attention_mask'],
            features['attention_mask'],
        ]

        # split to batches
        vsx_inputs = np.concatenate(vsx_inputs, axis=0)
        vsx_inputs = np.split(vsx_inputs, vsx_inputs.shape[0], axis=0)
        vsx_batches = []
        for i in range(len(vsx_inputs) // 6):
            vsx_batch = []
            for inp in vsx_inputs[i :: len(vsx_inputs) // 6]:
                vsx_batch.append(
                    vsx.from_numpy(
                        np.array(inp, dtype=np.int32),
                        self.device_id if hasattr(self, 'device_id') else 0,
                    )
                )
            vsx_batches.append(vsx_batch)
        
        outputs = vsx_engine(vsx_batches)
        return outputs

    def run_torch(self, torch_engine, features: Dict[str, np.ndarray]):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = torch.from_numpy(np.array(features['input_ids'], dtype=np.int32)).to(device)
        attention_mask = torch.from_numpy(np.array(features['attention_mask'], dtype=np.int32)).to(
            device
        )
        
        with torch.no_grad():
            outputs = torch_engine(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=False
            )

        return np.asarray(outputs[0].detach().cpu().numpy())

    def run_onnx(self, onnx_engine, features: Dict[str, np.ndarray]):
        input_ids = np.array(features['input_ids'], dtype=np.int32)
        attention_mask = np.array(features['attention_mask'], dtype=np.int32)
        token_type_ids = np.zeros(features['input_ids'].shape, dtype=np.int32)

        input_name = onnx_engine.get_inputs()[0].name
        output_name = onnx_engine.get_outputs()[0].name
        heatmap = onnx_engine.run([output_name], 
                                  {'input_ids': input_ids,
                                  'attention_mask': attention_mask,
                                  'token_type_ids': token_type_ids
                                  })[0]

        return heatmap

    def post_precess(self, outputs: np.ndarray) -> List:
        embeddings = outputs[:, 0]
        embeddings = torch.from_numpy(embeddings)
        # embeddings = torch.nn.functional.normalize(torch.from_numpy(embeddings), p=2, dim=1)
        # embeddings = torch.nn.functional.normalize( embeddings, p=2, dim=0)
        return embeddings.detach().cpu().numpy()


class Reranker(Embedding):
    def init_torch(self):
        from transformers import AutoModelForSequenceClassification

        torch_engine = (
            AutoModelForSequenceClassification.from_pretrained(self.torch_model_or_tokenizer)
            .eval()
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
        # print("june torch_engine ", torch_engine)
        return torch_engine

    def post_precess(self, outputs: np.ndarray, do_sigmod=False) -> List:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        embeddings = outputs[:, 0]
        
        if do_sigmod:
            score = sigmoid(embeddings)
        else:
            score = embeddings

        if score.shape == (2, 1):
            score = np.squeeze(score)

        return score
    
    def infer(self, engine, sentences: List[List[str]], do_sigmod=False) -> List:

        all_embeddings = []

        disable = False
        if len(sentences) < 3:
            # do not show process bar
            disable = True

        for start_index in trange(0, len(sentences), self.batch_size, desc="Rerank infer...", disable=disable):
            sentences_batch = sentences[start_index : start_index + self.batch_size]
            features = self.tokenizer(
                sentences_batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_seqlen,
                return_tensors="np",
            )
            if isinstance(engine, torch.nn.Module):
                token_embeddings = self.run_torch(engine, features)
            elif isinstance(engine, EmbeddingX):
                token_embeddings = self.run_vsx(engine, features)
            else:
                token_embeddings = self.run_onnx(engine, features)
            
            token_embeddings = self.post_precess(token_embeddings, do_sigmod=do_sigmod)

            all_embeddings.extend(token_embeddings)

        return np.array(all_embeddings)


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def evaluate_emb_sts12_sts(embedding_1, embedding_2, score_list, **kwargs):

    from eval_metric import STSEvaluator

    evaluator = STSEvaluator(embedding_1, embedding_2, score_list, min_score=0, max_score=5)
    eval_info = evaluator()

    return eval_info

def evaluate_emb_BQ(embedding_1, embedding_2, score_list, **kwargs):

    from eval_metric import STSEvaluator

    evaluator = STSEvaluator(embedding_1, embedding_2, score_list, min_score=0, max_score=1)
    eval_info = evaluator()

    return eval_info


def evaluate_rerank_MS(infer_data, mrr_at_k=10, **kwargs):
    from eval_metric import RerankingEvaluator

    evaluator = RerankingEvaluator(infer_data, mrr_at_k=mrr_at_k)
    eval_info = evaluator()

    return eval_info



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Deploy Embedding Model")
    parser.add_argument('--vacc_weight', type=str, default='vacc_deploy/bge-m3-512-int8-old-yaml-100/mod', help="model_name_or_path")
    parser.add_argument('--torch_weight', type=str, default='bge/bge-m3', help="pytorch weight for toknizer or infer")
    parser.add_argument('--onnx_weight', type=str, default='bge/bge-reranker-v2-m3/onnx/bge-reranker-v2-m3-512.onnx', help="onnx weight")
    parser.add_argument('--task', type=str, default='embedding', choices=["embedding", "reranker"])
    parser.add_argument('--eval_mode', action='store_true', help="eval or infer mode")
    parser.add_argument('--eval_engine', type=str, default='vacc', choices=["torch", "onnx", "vacc"], help="eval vacc model or torch moedl")
    parser.add_argument('--eval_dataset', type=str, default='jsonl/mteb-sts12-sts_test.jsonl', 
                        help="embedding task 'mteb-sts12-sts_test.jsonl' for english, 'c-mteb-bq_test.jsonl' for chinese, " 
                        "rerank task using 'zyznull-msmarco-passage-ranking_dev.jsonl, "
                        "download form: http://192.168.20.139:8888/vastml/dataset/text2vec/jsonl/")
    parser.add_argument('--seqlen', type=int, default=512, help="model seqlen")

    args = parser.parse_args()
    print(args)

    if args.task == 'embedding':

        demo = Embedding(vacc_model_prefix_path=args.vacc_weight,
                         torch_model_or_tokenizer=args.torch_weight,
                         onnx_model_path=args.onnx_weight,
                         device_id=1,
                         batch_size=1,
                         max_seqlen=args.seqlen,
                         )
        
        if not args.eval_mode:
            sentences = [
                'Hybrid retrieval leverages the strengths of various methods',
                '它同时支持 embedding 和稀疏检索，这样在生成稠密 embedding 时，无需付出额外代价，就能获得与 BM25 类似的 token 权重',
                '我们建议使用以下流程：混合检索+重新排名',
                "测试句子好不好用，因为我做了修改你知道吗",
            ]
            vsx_engine = demo.init_vsx()
            torch_engine = demo.init_torch()

            vsx_embeds = demo.infer(vsx_engine, sentences)
            torch_embeds = demo.infer(torch_engine, sentences)
            logger.info(f"torch vs vacc, embedding cosine_similarity: {cosine_similarity(vsx_embeds, torch_embeds)}")
            vsx_engine.finish()
        else:
            if 'sts12-sts' in args.eval_dataset or 'c-mteb-bq' in args.eval_dataset:
                dataset = datasets.load_dataset('json', data_files=args.eval_dataset, split='train')
            else:
                raise ValueError("in embedding task, only support eval dataset in ['sts12-sts_test.jsonl', 'c-mteb-bq_test.jsonl']")
            
            sentences1= dataset['sentence1']
            sentences2 = dataset['sentence2']
            gold_scores = dataset['score']

            if args.eval_engine == "torch":
                engine = demo.init_torch()
            elif args.eval_engine == "onnx":
                engine = demo.init_onnx()
            else:
                engine = demo.init_vsx()  

            embeds1 = demo.infer(engine, sentences1)
            embeds2 = demo.infer(engine, sentences2)

            if 'sts12-sts' in args.eval_dataset:
                eval_info = evaluate_emb_sts12_sts(embeds1, embeds2, gold_scores)
            else:
                eval_info = evaluate_emb_BQ(embeds1, embeds2, gold_scores)

            logger.info(f"embedding eval info: {eval_info}")
            if args.eval_engine == "vacc":
                engine.finish()
    else:
        demo = Reranker(
                vacc_model_prefix_path=args.vacc_weight,
                torch_model_or_tokenizer=args.torch_weight,
                onnx_model_path=args.onnx_weight,
                device_id=1,
                batch_size=8,
                max_seqlen=args.seqlen,
            )
        
        if not args.eval_mode:
            sentences = [['what is panda?', 'hi'],
                        ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

            vsx_engine = demo.init_vsx()
            torch_engine = demo.init_torch()
            
            # score convert to [0,1]
            do_sigmod = True

            torch_embeds = demo.infer(torch_engine, sentences, do_sigmod=do_sigmod)
            logger.info(f"torch reank: {torch_embeds}")

            vsx_embeds = demo.infer(vsx_engine, sentences, do_sigmod=do_sigmod)
            logger.info(f"vacc reank: {vsx_embeds}")

            vsx_engine.finish()
        else:
            if 'msmarco-passage-ranking' in args.eval_dataset:
                dataset = datasets.load_dataset('json', data_files=args.eval_dataset, split='train[:100]')
            else:
                raise ValueError("in reranker task, only support eval dataset in 'msmarco-passage-ranking_dev.jsonl'")
            
            if args.eval_engine == "torch":
                engine = demo.init_torch()
            elif args.eval_engine == "onnx":
                engine = demo.init_onnx()
            else:
                engine = demo.init_vsx() 
            
            # score convert to [0,1]
            do_sigmod = True

            mrr_at_k = 10
            infer_data = []
            for data in tqdm(dataset, colour='blue'):
                if min(len(data['positive_passages']), len(data['negative_passages'])) < 1:
                    continue
                if len(data['negative_passages']) < mrr_at_k:
                    continue

                sentences_query = data['query']
                sentences_query_id = data['query_id']
                sentences_positive = data['positive_passages']
                sentences_negative = data['negative_passages']
                
                query_positive = [[sentences_query, sentences_positive[0]['text']]] # positive only one text
                query_negative = [[sentences_query, sentences_negative[i]['text']] for i in range(len(sentences_negative))] # negative have 29 text

                positive_score = []
                for sentences in query_positive:
                    score = demo.infer(engine, [sentences], do_sigmod=do_sigmod)
                    positive_score.append(score)
                
                negative_score = []  
                for sentences in query_negative:
                    score = demo.infer(engine, [sentences], do_sigmod=do_sigmod)
                    negative_score.append(score)

                infer_data.append({'id': sentences_query_id,
                                   'query_positive': query_positive,
                                   'query_negative': query_negative,
                                   'query_positive_score': positive_score,
                                   'query_negative_score': negative_score})

            eval_info = evaluate_rerank_MS(infer_data, mrr_at_k)
            logger.info(f"reranker eval info: {eval_info}")
            if args.eval_engine == "vacc":
                engine.finish()
    


'''
sts12-sts_test.jsonl

bge-m3-512-fp32-torch
[09/20/2024-21:21:53] embedding eval info: {'pearson': 0.837127502430422, 'spearman': 0.787349619213163, 'cosine_pearson': 0.8371275011085642, 'cosine_spearman': 0.7873426613663244, 'manhattan_pearson': 0.8037256147152081, 'manhattan_spearman': 0.7915637276660088, 'euclidean_pearson': 0.803910823016911, 'euclidean_spearman': 0.7918675408778761}

bge-m3-512-fp32-onnx
[03/05/2025-15:33:44] embedding eval info: {'pearson': 0.837127509735718, 'spearman': 0.7873292270734319, 'cosine_pearson': 0.8371275064421444, 'cosine_spearman': 0.7873428069462605, 'manhattan_pearson': 0.8037256323320123, 'manhattan_spearman': 0.791563285267859, 'euclidean_pearson': 0.8039108431048845, 'euclidean_spearman': 0.791867863650916}

bge-m3-512-fp16-vacc
[09/20/2024-21:02:29] torch vs vacc, embedding cosine_similarity: 0.9999426007270813

[09/20/2024-20:51:42] embedding eval info: {'pearson': 0.8372050635377709, 'spearman': 0.7873990509404183, 'cosine_pearson': 0.8372163163355593, 'cosine_spearman': 0.7873534120211125, 'manhattan_pearson': 0.8038242956831485, 'manhattan_spearman': 0.7916207918075133, 'euclidean_pearson': 0.8040076446650469, 'euclidean_spearman': 0.7919682273871113}

bge-m3-512-int8-vacc 混精
[03/26/2025-14:11:59] embedding eval info: {'pearson': 0.8370460679976922, 'spearman': 0.7872192737434466, 'cosine_pearson': 0.8380377389416677, 'cosine_spearman': 0.7881945236136848, 'manhattan_pearson': 0.8048959994601577, 'manhattan_spearman': 0.7924705372148708, 'euclidean_pearson': 0.80508816184137, 'euclidean_spearman': 0.7927361870144013}

'''





'''
msmarco-passage-ranking_dev.jsonl 100

512-torch
[09/22/2024-15:57:30] reranker eval info: {'map': 0.5789126982206567}

512-onnx
[03/05/2025-16:17:41] reranker eval info: {'map': 0.5789126982206567}

bge-reranker-v2-m3-512-fp16-nolayout
[09/23/2024-10:07:48] reranker eval info: {'map': 0.5147646459046933}

bge-reranker-v2-m3-512-int8-混精
[03/26/2025-11:55:05] reranker eval info: {'map': 0.49322842190489247}
'''
