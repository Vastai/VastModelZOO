# ==============================================================================
# 
# Copyright (C) 2023 VastaiTech Technologies Inc.  All rights reserved.
# 
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :     vastml
@Time  : 2023/04/07 22:08:31
'''

import argparse
import thop
import torch
from thop import clever_format
# from models.build_model import build_model
from models.iresnet import iResNet,Bottleneck


class select_fun():

    def iresnet50(root,**kwargs):
        """Constructs a iResNet-50 model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = iResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        model.load_state_dict(torch.load(root+"iresnet50.pth"))

        return model


    def iresnet101(root,**kwargs):
        """Constructs a iResNet-101 model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = iResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
        model.load_state_dict(torch.load(root+"iresnet101.pth")) 
        return model


    def iresnet152(root,**kwargs):
        """Constructs a iResNet-152 model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = iResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
        model.load_state_dict(torch.load(root+"iresnet152.pth")) 

        return model


    def iresnet200(root,**kwargs):
        """Constructs a iResNet-200 model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = iResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
        model.load_state_dict(torch.load(root+"iresnet200.pth")) 

        return model

def parse_args():
    parser = argparse.ArgumentParser(description='Convert pretrained model to onnx file.')
    parser.add_argument('--model_name',default="iresnet50",help="iresnet50、iresnet101、iresnet152、iresnet200")
    parser.add_argument('--infile',default='/home/model/pretrained/',help="folder of pre-trained model")
    parser.add_argument('--output',default='/home/model/onnx',help="path to save onnx file")
    args = parser.parse_args()
    return args

def count_op(model, input):
    flops, params = thop.profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

def main(args):
    model_fun = getattr(select_fun,args.model_name)
    model = model_fun(args.infile)
    img = torch.randn(1,3,224,224)
    model.eval()
    count_op(model, img)
    torch.onnx.export(model,img,args.output+args.model_name+".onnx", input_names=["input"], opset_version=10)
    print(f"{args.model_name} export done")

if __name__ == "__main__":

    args = parse_args()
    main(args)