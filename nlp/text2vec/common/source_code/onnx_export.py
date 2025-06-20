# ==============================================================================
#
# Copyright (C) 2024 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :          lance
@Email : lance.wang@vastaitech.com
@Time  : 	2025/04/21 19:43:31
'''

import os
import shutil
import torch
import argparse
import transformers
import transformers.models

import onnx
import onnxruntime
from onnxsim import simplify

import numpy as np

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_info(model, tokenizer, device: str = "cpu"):
    # ops info
    from modelops_analyzer import torch_model

    inputs = tokenizer(
        'test model info',
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

    torch_model.Ops(
        model=model,
        details=True,
        example_inputs=(inputs_on_device['input_ids'], inputs_on_device['attention_mask']),
    ).summary()

    torch_model.summary(
        model, input_data=(inputs_on_device['input_ids'], inputs_on_device['attention_mask'])
    )


def forward(
    self,
    input_ids=None,
    token_type_ids=None,
    position_ids=None,
    inputs_embeds=None,
    past_key_values_length=0,
):
    if position_ids is None:
        position_ids = torch.Tensor(
            [[i for i in range(input_ids.shape[1])] for j in range(input_ids.shape[0])]
        ).to(dtype=input_ids.dtype)
    if self.__class__ is transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaEmbeddings:
        self.position_embeddings.weight = torch.nn.Parameter(
            self.position_embeddings.weight[self.padding_idx:]
        )
    else:
        self.position_embeddings.weight = torch.nn.Parameter(
        self.position_embeddings.weight
    )
    print("Patch Position_ids :", position_ids)
    print("Patch Position Embed :", self.position_embeddings.weight.shape)
    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]

    self.max_length = input_shape[1]

    # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
    # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
    # issue #5664
    if token_type_ids is None:
        if hasattr(self, "token_type_ids"):
            buffered_token_type_ids = self.token_type_ids[:, : self.max_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                input_shape[0], self.max_length
            )
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = inputs_embeds + token_type_embeddings
    if self.position_embedding_type == "absolute":
        position_embeddings = self.position_embeddings(position_ids.to(device))
        embeddings += position_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


class Embedding:
    def __init__(
        self,
        model_name_or_path,
        mode="embedding",
        do_patch=False,
        max_length: int = 512,
    ) -> None:
        assert mode in ["embedding", "reranker"]
        self.model_name = model_name_or_path.split('/')[-1]
        if self.model_name == '':
            self.model_name = model_name_or_path.split('/')[-2] 
        self.mode = mode
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print(f"Model Max Seqlen is {self.tokenizer.model_max_length}, set is {self.max_length}")
        assert self.max_length <= self.tokenizer.model_max_length
        if mode == "reranker":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path
            ).eval()
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path).eval()
            # modify
            self.model.pooler = None
        self.model.to(device)
        if do_patch:
            if mode == "embedding":
                if type(self.model.embeddings) == transformers.models.bert.modeling_bert.BertEmbeddings:
                    transformers.models.bert.modeling_bert.BertEmbeddings.forward = (
                        forward
                    )
                else:
                    transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaEmbeddings.forward = (
                        forward
                    )
            else:
                # bert
                if type(self.model) == transformers.models.bert.modeling_bert.BertForSequenceClassification:
                    transformers.models.bert.modeling_bert.BertEmbeddings.forward = (
                        forward
                    )
                    print("bert bert bert bert")
                else:
                    #roberta
                    transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaEmbeddings.forward = (
                        forward
                    )
                    print("xlm_roberta xlm_roberta xlm_roberta xlm_roberta")
                    #  self.model.roberta.embeddings) == transformers.models.bert.modeling_bert.BertEmbeddings
                
            print("*********************\n Patch Model Position IDS Done \n*********************")

    def run(
        self,
        sentence_batch=["测试句子好不好用，因为我做了修改你知道吗", "检查句子是否正确 how are you"],
    ):
        inputs = self.tokenizer(
            sentence_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs_on_device, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        print("HF Sentence embeddings:", embeddings.shape)
        return embeddings

    def run_reranker(
        self,
        pairs=[
            ['what is panda?', 'hi'],
            [
                'what is panda?',
                'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.',
            ],
        ],
    ):

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding='max_length', truncation=True, return_tensors='pt'
            ).to(device)
            print("-------- \n model input shape:", inputs['input_ids'].shape, "\n--------")
            output = self.model(**inputs, return_dict=True)
            scores = output.logits.view(
                -1,
            ).float()

            scores = scores.cpu().numpy().tolist()

            # normal
            normal_scores = [sigmoid(score) for score in scores]
            print(f"Ranker Score: {scores}, Normal ranker Score: {normal_scores}")

            return scores

    def export(self):
        # dummy run
        input = torch.randint(0, 1, (1, self.max_length), dtype=torch.int32).to(device)

        save_onnx_dir = os.path.join(args.save_dir, self.model_name, "temp")
        save_onnx_file = os.path.join(save_onnx_dir, self.model_name + f'-{self.max_length}.onnx')
        os.makedirs(save_onnx_dir, exist_ok=True)
        # export
        torch.onnx.export(
            self.model,
            (input, input, input),
            save_onnx_file,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            opset_version=11,
            # dynamic_axes={"input_ids":{0:"batch"}, "attention_mask":{0:"batch"}, "token_type_ids":{0:"batch"} },
        )

        # # simpliy
        model = onnx.load(save_onnx_file)
        simplify_model, _ = simplify(model)
        save_onnx_sim_file = os.path.join(args.save_dir, self.model_name, self.model_name + f'-{self.max_length}-sim.onnx')
        onnx.save(simplify_model, save_onnx_sim_file) # , save_as_external_data=True, all_tensors_to_one_file=True

        shutil.rmtree(save_onnx_dir)
        # onnxruntime
        onnx_session = onnxruntime.InferenceSession(save_onnx_sim_file)
        input_name_0 = onnx_session.get_inputs()[0].name
        input_name_1 = onnx_session.get_inputs()[1].name
        input_name_2 = onnx_session.get_inputs()[2].name

        heatmap = onnx_session.run(None, {input_name_0: np.array(input),
                                          input_name_1: np.array(input),
                                          input_name_2: np.array(input)})[0]

        print(heatmap.shape)

if __name__ == "__main__":
    # Embedding('bge-reranker-v2-m3', 'reranker', False).run_reranker()
    # exit()
    # Embedding('bge-m3', 'embedding', True).run()
    # print(torch.cosine_similarity(Embedding(model_name_or_path, reranker, do_patch=False).run(), Embedding(model_name_or_path, reranker, do_patch=True).run()))
    # exit()
    parser = argparse.ArgumentParser(description="Deploy Embedding Model")
    parser.add_argument('--model', type=str, default='./vacc_deploy/bge-reranker-large', help="model_name_or_path")
    parser.add_argument('--type', type=str, default='reranker', choices=["embedding", "reranker"])
    parser.add_argument('--patch', type=bool, default=True, help="patch forward for vacc")
    parser.add_argument('--seqlen', type=int, default=512, help="model seqlen")
    parser.add_argument('--save_dir', type=str, default='./vacc_deploy/onnx_weights', help="onnx save folder")

    args = parser.parse_args()
    print(args)

    if args.type == "embedding":
        Embedding(
            model_name_or_path=args.model,
            mode="embedding",
            do_patch=args.patch,
            max_length=args.seqlen,
        ).export()
    else:
        Embedding(
            model_name_or_path=args.model,
            mode="reranker",
            do_patch=args.patch,
            max_length=args.seqlen,
        ).export()