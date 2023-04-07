'''
Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.

The information contained herein is confidential property of the company.
The user, copying, transfer or disclosure of such information is prohibited
except by express written agreement with VASTAI Technologies Co., Ltd.
'''
import argparse
import os

import thop
import torch
from thop import clever_format

parse = argparse.ArgumentParser(description="MAKE MODELS CLS FOR VACC")
parse.add_argument("--model_library", type=str, default="timm", choices=["timm", "torchvision"])
parse.add_argument("--model_name", type=str, default="dpn107")
parse.add_argument("--save_dir", type=str, default="../../weights/efficientnet/")
parse.add_argument("--size", type=int, default=224)
parse.add_argument(
    "--pretrained_weights",
    type=str,
    # default="../../weights/efficientnet/tf_efficientnet_b8.pth",
    default=None,

    help="timm or torchvision or custom onnx weights path",
)
parse.add_argument(
    "--convert_mode",
    type=str,
    default="onnx",
    choices=["onnx", "pt"],
)
args = parse.parse_args()
print(args)


class ModelHUb:
    def __init__(self, opt):
        self.model_name = opt.model_name
        self.pretrained_weights = opt.pretrained_weights
        self.convert_mode = opt.convert_mode
        self.num_class = 1000
        self.img = torch.randn(1, 3, opt.size, opt.size)

        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        self.save_file = os.path.join(opt.save_dir, self.model_name + "_" + str(opt.size) + "." + self.convert_mode)
        if opt.model_library == "timm":
            self.model = self._get_model_timm()
        else:
            self.model = self._get_model_torchvision()

        count_op(self.model, self.img)

    def get_model(self):
        if self.convert_mode == "onnx":
            torch.onnx.export(self.model, self.img, self.save_file, input_names=["input"], opset_version=11)
        else:
            self.model(self.img)  # dry runs
            scripted_model = torch.jit.trace(self.model, self.img, strict=False)
            torch.jit.save(scripted_model, self.save_file)
        print("[INFO] convert model save:", self.save_file)

    def _get_model_torchvision(self):
        """通过torchvision加载预训练模型"""
        import torchvision

        if self.pretrained_weights:
            model = torchvision.models.__dict__[self.model_name](pretrained=False, num_classes=self.num_class)
            checkpoint = torch.load(self.pretrained_weights)
            model.load_state_dict(checkpoint)
        else:
            model = torchvision.models.__dict__[self.model_name](pretrained=True)
        model.eval()
        return model

    def _get_model_timm(self):
        """通过timm加载预训练模型"""
        import timm

        if self.pretrained_weights:
            model = timm.create_model(
                model_name=self.model_name,
                num_classes=self.num_class,
                pretrained=False,
                checkpoint_path=self.pretrained_weights,
            )
        else:
            model = timm.create_model(
                model_name=self.model_name,
                num_classes=self.num_class,
                pretrained=True,
            )
        model.eval()
        return model


def count_op(model, input):
    flops, params = thop.profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)


if __name__ == "__main__":
    """
        此代码用来转换timm和torchvision中模型
    """
    maker = ModelHUb(args)
    maker.get_model()