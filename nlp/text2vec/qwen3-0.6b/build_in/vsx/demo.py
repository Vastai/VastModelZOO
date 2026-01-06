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
from torch import Tensor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%m/%d/%Y-%H:%M:%S",
    handlers=[logging.NullHandler(), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def cosine_similarity(A, B):
    return np.dot(A.flatten(), B.flatten()) / (np.linalg.norm(A) * np.linalg.norm(B))

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output

class Embedding:
    def __init__(
        self,
        vacc_model_prefix_path: Union[str, Dict[str, str]],
        torch_model_or_tokenizer: str,
        device_id: int = 0,
        batch_size: int = 4,
        max_seqlen: int = 512,
        eval_mode: bool = False,
    ):

        self.max_seqlen = max_seqlen
        self.torch_model_or_tokenizer = torch_model_or_tokenizer
        self.batch_size = batch_size
        self.device_id = device_id
        self.vacc_model_prefix_path = vacc_model_prefix_path
        self.eval_mode = eval_mode
        # init tokenizer
        self.tokenizer_torch = AutoTokenizer.from_pretrained(
            self.torch_model_or_tokenizer, trust_remote_code=True, padding_side='left'
        )
        self.tokenizer_vacc = AutoTokenizer.from_pretrained(
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
        return torch_engine

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

        disable = False
        if len(sentences) < 3:
            # do not show process bar
            disable = True
        
        # Each query must come with a one-sentence instruction that describes the task
        task = 'Retrieve semantically similar text'

        for start_index in trange(0, len(sentences), self.batch_size, desc="Batches", disable=disable):
            sentences_batch = sentences[start_index : start_index + self.batch_size]
            if self.eval_mode:
                sentences_batch= [get_detailed_instruct(task, sentence) for sentence in sentences_batch]
            
            features = self.tokenizer_vacc(
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
        if self.eval_mode:
            embeddings = last_token_pool(torch.tensor(outputs), torch.from_numpy(np.array(features['attention_mask'], dtype=np.int32)))
            return np.asarray(embeddings.detach().cpu().numpy())
        else:
            return outputs

    def run_torch(self, torch_engine, features: Dict[str, np.ndarray]):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = torch.from_numpy(np.array(features['input_ids'], dtype=np.int32)).to(device)
        attention_mask = torch.from_numpy(np.array(features['attention_mask'], dtype=np.int32)).to(
            device
        )
        if self.eval_mode:
            with torch.no_grad():
                outputs = torch_engine(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=False
                )
                embeddings = last_token_pool(outputs[0], attention_mask)
            return np.asarray(embeddings.detach().cpu().numpy())
        else:
            with torch.no_grad():
                outputs = torch_engine(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=False
                )
            return np.asarray(outputs[0].detach().cpu().numpy())

    def post_precess(self, outputs: np.ndarray) -> List:
        if not args.eval_mode:
            outputs = outputs[:, 0]
        embeddings = torch.from_numpy(outputs)
        embeddings = F.normalize(embeddings, p=2, dim=0)
        return embeddings.detach().cpu().numpy()


class Reranker(Embedding):
    
    def init_torch(self):
        from transformers import AutoModelForCausalLM

        torch_engine = (
            AutoModelForCausalLM.from_pretrained(self.torch_model_or_tokenizer)
            .eval()
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )

        return torch_engine
    def _select_max_seq_len(self, max_input_seq_len) -> int:
        if self.max_seqlen >= max_input_seq_len:
            return self.max_seqlen
        raise ValueError(
            f"Input sequence length {max_input_seq_len} exceeds maximum supported length {self.max_seqlen}"
        )
    def process_inputs(self, pairs, prefix_tokens, suffix_tokens, tokenizer):
        inputs = tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_seqlen - len(prefix_tokens) - len(suffix_tokens)
        )
        input_ids_len_arr = []
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
            input_ids_len_arr.append(len(inputs["input_ids"][i]))
        max_input_seq_len = max(input_ids_len_arr)
        max_seqlen = 0
        try:
            max_seqlen = self._select_max_seq_len(max_input_seq_len)
        except ValueError as e:
            logger.error(str(e))
            raise e
        inputs = tokenizer.pad(inputs, padding="max_length", return_tensors="np", max_length=max_seqlen)

        return inputs, input_ids_len_arr

    def post_precess(self, outputs, token_true_id, token_false_id) -> List:
        batch_scores = outputs[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def post_precess_vsx(self, outputs, token_true_id, token_false_id, input_ids_len_arr):
        outputs = torch.tensor(outputs)
        scores = []
        vocab_size = 151669
        for index, val in enumerate(input_ids_len_arr):
            batch_scores = outputs[index, val - 1, :vocab_size]
            true_vector = batch_scores[token_true_id]
            false_vector = batch_scores[token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=0)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=0)
            score = batch_scores[1].exp().item()
            scores.append(score)

        return scores

    def infer(self, engine, sentences: List[List[str]]) -> List:

        all_embeddings = []

        disable = False
        if len(sentences) < 3:
            # do not show process bar
            disable = True

        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if isinstance(engine, EmbeddingX):
            tokenizer = self.tokenizer_vacc
        else:
            tokenizer = self.tokenizer_torch
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")

        task = "Given a web search query, retrieve relevant passages that answer the query"

        for start_index in trange(0, len(sentences), self.batch_size, desc="Rerank infer...", disable=disable):
            sentences_batch = sentences[start_index : start_index + self.batch_size]
            sentences_batch = [format_instruction(task, sentence[0], sentence[1]) for sentence in sentences_batch]

            inputs, input_ids_len_arr = self.process_inputs(sentences_batch, prefix_tokens, suffix_tokens, tokenizer)
            
            if isinstance(engine, torch.nn.Module):
                token_embeddings = self.run_torch(engine, inputs)
                token_embeddings = self.post_precess(token_embeddings, token_true_id, token_false_id)
            elif isinstance(engine, EmbeddingX):
                token_embeddings = self.run_vsx(engine, inputs)
                token_embeddings = self.post_precess_vsx(token_embeddings, token_true_id, token_false_id, input_ids_len_arr)

            all_embeddings.extend(token_embeddings)

        return np.array(all_embeddings)

    def run_vsx(self, vsx_engine, features):

        vsx_inputs = [
            features["input_ids"],
            features["attention_mask"],
            features["attention_mask"],
            features["attention_mask"],
            features["attention_mask"],
            features["attention_mask"],
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

    def run_torch(self, torch_engine, inputs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = torch.from_numpy(np.array(inputs['input_ids'], dtype=np.int32)).to(device)
        attention_mask = torch.from_numpy(np.array(inputs['attention_mask'], dtype=np.int32)).to(
            device
        )
        with torch.no_grad():
            outputs = torch_engine(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        return outputs[0]


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
    parser.add_argument('--vacc_weight', type=str, default='vacc_deploy/Qwen3-Embedding-0.6B-VACC-512/prefill_512_rank0/mod', help="model_name_or_path")
    parser.add_argument('--torch_weight', type=str, default='Qwen3-Embedding-0.6B', help="pytorch weight for toknizer or infer")
    parser.add_argument('--task', type=str, default='embedding', choices=["embedding", "reranker"])
    parser.add_argument('--eval_mode', action='store_true', help="eval or infer mode")
    parser.add_argument('--eval_engine', type=str, default='vacc', choices=["torch", "vacc"], help="eval vacc model or torch model")
    parser.add_argument('--eval_dataset', type=str, default='jsonl/mteb-sts12-sts_test.jsonl', 
                        help="embedding task 'mteb-sts12-sts_test.jsonl' for english, 'c-mteb-bq_test.jsonl' for chinese, " 
                        "rerank task using 'zyznull-msmarco-passage-ranking_dev.jsonl, ")
    parser.add_argument('--seqlen', type=int, default=512, help="model seqlen")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--device_id', type=int, default=0, help="device id")

    args = parser.parse_args()
    print(args)

    if args.task == 'embedding':

        demo = Embedding(vacc_model_prefix_path=args.vacc_weight,
                         torch_model_or_tokenizer=args.torch_weight,
                         device_id=args.device_id,
                         batch_size=args.batch_size,
                         max_seqlen=args.seqlen,
                         eval_mode=args.eval_mode,
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
                device_id=args.device_id,
                batch_size=args.batch_size,
                max_seqlen=args.seqlen,
                eval_mode=args.eval_mode,
            )
        
        if not args.eval_mode:
            sentences = [["What is the capital of China?","The capital of China is Beijing.",],
                        ['Explain gravity', 'Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.'],
                        ['what is panda?', 'hi'],
                        ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

            vsx_engine = demo.init_vsx()
            torch_engine = demo.init_torch()
        

            torch_embeds = demo.infer(torch_engine, sentences)
            logger.info(f"torch reank: {torch_embeds}")

            vsx_embeds = demo.infer(vsx_engine, sentences)
            logger.info(f"vacc reank: {vsx_embeds}")

            vsx_engine.finish()
        else:
            if 'msmarco-passage-ranking' in args.eval_dataset:
                dataset = datasets.load_dataset('json', data_files=args.eval_dataset, split='train[:100]')
            else:
                raise ValueError("in reranker task, only support eval dataset in 'msmarco-passage-ranking_dev.jsonl'")
            
            if args.eval_engine == "torch":
                engine = demo.init_torch()
            else:
                engine = demo.init_vsx() 

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
                    score = demo.infer(engine, [sentences])
                    
                    positive_score.append(score)
                
                negative_score = []  
                for sentences in query_negative:
                    score = demo.infer(engine, [sentences])
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
    
