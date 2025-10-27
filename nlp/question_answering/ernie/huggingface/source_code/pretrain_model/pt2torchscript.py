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
from transformers import AutoModelForQuestionAnswering


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="FINETUNE MODEL")
    parse.add_argument(
        "--model_name_or_path", 
        type=str,
        default="./project/nlp/weights/finetune/qa/ernie2.0-base-en",
        help="pretrain model path"
    )
    parse.add_argument(
        "--seq_length", 
        type=int, 
        default=384, 
        help="max sequence length "
    )
    parse.add_argument(
        "--save_path",
        type=str,
        default="torchscript/ernie2.0-base-en.torchscript.pt",
        help="finetune model result dir.",
    )
    args = parse.parse_args()
    
    model_name_or_path = args.model_name_or_path
    seq_length = args.seq_length
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path, return_dict=False)
    model.eval()
    model.export = True
    # model = modify_bert_embeddings(model, seq_length)
    input = torch.zeros((1, seq_length), dtype=torch.long)
    out = model(input, input, input)
    print(out.shape)
    
    scripted_model = torch.jit.trace(model, (input, input, input) , strict=False).eval()
    torch.jit.save(scripted_model, args.save_path)
        



