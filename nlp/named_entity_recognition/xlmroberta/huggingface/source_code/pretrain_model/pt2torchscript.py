# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import argparse
import sys
sys.path.append("..")

import torch
from transformers import AutoModelForTokenClassification
from finetune.xlm_roberta_embeddings import modify_xlm_roberta_embeddings


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="FINETUNE MODEL")
    parse.add_argument(
        "--model_name_or_path", 
        type=str,
        default="./code/nlp/transformers/runs/Pople_for_ner_xmlroberta_base_zh/checkpoint-13040",
        help="pretrain model path"
    )
    parse.add_argument(
        "--seq_length", 
        type=int, 
        default=256, 
        help="max sequence length "
    )
    parse.add_argument(
        "--save_path",
        type=str,
        default="./code/nlp/transformers/runs/Pople_for_ner_xmlroberta_base_zh/checkpoint-13040/ner_xlmroberta_base_en-256.torchscript.pt",
        help="finetune model result dir.",
    )
    args = parse.parse_args()
    
    model_name_or_path = args.model_name_or_path
    seq_length = args.seq_length
    
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, return_dict=False)
    model = modify_xlm_roberta_embeddings(model, seq_length)
    input = torch.randint(0, 1, (1, seq_length), dtype=torch.long)
    out = model(input, input, input)
    print(out[0].shape)
    
    model.eval()
    scripted_model = torch.jit.trace(model, (input, input, input) , strict=False).eval()
    torch.jit.save(scripted_model, args.save_path)



