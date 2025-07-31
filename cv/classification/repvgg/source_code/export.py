# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import argparse
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from repvgg import get_RepVGG_func_by_name, repvgg_model_convert

parser = argparse.ArgumentParser(description='RepVGG Conversion')
parser.add_argument('load', metavar='LOAD', help='path to the weights file')
parser.add_argument('save', metavar='SAVE', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')
parser.add_argument('-m', '--mode', default='onnx',choices=["onnx","torchscript"])

def convert():
    args = parser.parse_args()
    if 'plus' in args.arch:
        from repvggplus import get_RepVGGplus_func_by_name
        train_model = get_RepVGGplus_func_by_name(args.arch)(deploy=False, use_checkpoint=False)
    else:
        repvgg_build_func = get_RepVGG_func_by_name(args.arch)
        train_model = repvgg_build_func(deploy=False)

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        print(ckpt.keys())
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    if 'plus' in args.arch:
        train_model.switch_repvggplus_to_deploy()
        torch.save(train_model.state_dict(), args.save)
    else:
        deploy_model = repvgg_model_convert(train_model, save_path=args.save)
        img = torch.randn(1,3,224,224)
        if args.mode == "onnx" :
            torch.onnx.export(deploy_model,img,args.save.replace('.pth','.onnx'),input_names=["input"], opset_version=10)
        else:
            deploy_model(img)  # dry runs
            scripted_model = torch.jit.trace(deploy_model, img, strict=False)
            torch.jit.save(scripted_model, args.save.replace('.pth','_torchscript.pt'))






if __name__ == "__main__":
    convert()
