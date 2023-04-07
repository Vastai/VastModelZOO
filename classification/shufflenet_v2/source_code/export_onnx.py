'''
Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.

The information contained herein is confidential property of the company.
The user, copying, transfer or disclosure of such information is prohibited
except by express written agreement with VASTAI Technologies Co., Ltd.
'''
from collections import OrderedDict
import torch
import argparse
from network import ShuffleNetV2

def parse_args():
    parser = argparse.ArgumentParser(description="Convert front model to vacc.")
    parser.add_argument('--modelsize',default='0.5x',help='0.5x, 1.0x, 1.5x, 2.0x')
    args = parser.parse_args()
    return args

model_dict = {
        '0.5x':"./weights/ShuffleNetV2.0.5x.pth.tar",
        '1.0x':"./weights/ShuffleNetV2.1.0x.pth.tar",
        '1.5x':"./weights/ShuffleNetV2.1.5x.pth.tar",
        '2.0x':"./weights/ShuffleNetV2.2.0x.pth.tar"
    }

def export(model_size,weights):
    img = torch.randn(1, 3, 224,224)
    model = ShuffleNetV2(model_size=model_size)
    new_state_dict = OrderedDict()
    print(f"---------------------------{weights}--------------------------------")
    checkpoint = torch.load(weights, map_location="cpu")
    # model.load_state_dict(checkpoint['state_dict'], strict=True)
    checkpoint = dict(checkpoint['state_dict'])
    for k, v in checkpoint.items():
        name = k.replace('module.','') # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    print("-----------------load_checkpoint successfully---------------------------")
    torch.onnx.export(model, img, f"./weights/shufflenetv2_{model_size}.onnx", input_names=["input"], opset_version=10)
    print("export onnx successfully")
    model(img)  # dry runs
    scripted_model = torch.jit.trace(model, img, strict=True)
    torch.jit.save(scripted_model, f"./weights/shufflenetv2_{model_size}.torchscript.pt")
    print("export  torchscript successfully")

if __name__ == "__main__":
    args = parse_args()
    export(args.modelsize,model_dict[args.modelsize])