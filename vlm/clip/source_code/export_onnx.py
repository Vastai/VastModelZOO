# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import clip
from PIL import Image
import argparse
import onnx
import torch
from onnxsim import simplify
import os

def simplify_onnx(args):

    onnx_model = onnx.load_model(args.onnx_file)
    print(f"load onnx model from {args.onnx_file} to simplify.")

    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(
        model_simp,
        args.onnx_file.replace(".onnx", "_sim.onnx")
    )
    print("onnx save successed.")

def export_onnx(args):
    jit_model, transform = clip.load(model_name, device=device, jit=False)
    # onnx inputs
    text = clip.tokenize(["a diagram"]).to(device)
    image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)
    if args.export == "image":
        torch.onnx.export(
            jit_model,
            (image),
            args.onnx_file,
            input_names=["image"],
        )
    else:
        torch.onnx.export(
            jit_model,
            (text),
            args.onnx_file,
            input_names=["text"],
        )

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="EXPORT CLIP ONNX")
    parse.add_argument(
        "--onnx_file",
        type=str,
        default="onnx/images.onnx",
        help="path to save onnx files",
    )
    parse.add_argument("--model_name",
        type=str,
        default="ViT-B/32",
        help="path to train config"
    )
    parse.add_argument(
        "--export",
        type=str,
        choices=["image", "text"],
        help="choose backbone to export"
    )
    args = parse.parse_args()
    # fixed args
    device = "cpu"
    model_name = args.model_name
    if not os.path.exists(os.path.dirname(args.onnx_file)):
        os.makedirs(os.path.dirname(args.onnx_file))

    export_onnx(args)
    simplify_onnx(args)
