# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import argparse

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="FINETUNE MODEL")
    parse.add_argument(
        "--model_name_or_path", 
        type=str,
        default="./project/nlp/finetune_output/albert_tiny/checkpoint-345",
        help="pretrain model path"
    )
    parse.add_argument(
        "--seq_length", 
        type=int, 
        default=128, 
        help="max sequence length "
    )
    parse.add_argument(
        "--save_path",
        type=str,
        default="torchscript/cls_albert_tiny.torchscript.pt",
        help="finetune model result dir.",
    )
    args = parse.parse_args()
    
    model_name_or_path = args.model_name_or_path
    seq_length = args.seq_length
    
    model =  AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=False)
    input = torch.randint(0, 2, (1, seq_length), dtype=torch.long)
    out = model(input, input, input)
    print(out[0].shape)
    
    model.eval()
    scripted_model = torch.jit.trace(model, (input, input, input) , strict=False).eval()
    torch.jit.save(scripted_model, args.save_path)



