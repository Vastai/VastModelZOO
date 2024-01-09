import os
import sys
import torch
import importlib
import argparse
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lesrcnn") # do not modify
    parser.add_argument("--ckpt_path", type=str, default="x2/lesrcnn_x2.pth")
    parser.add_argument("--group", type=int, default=1) # do not modify
    parser.add_argument("--scale", type=int, default=2)
    return parser.parse_args()


def export(cfg):

    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + 'x{}'.format(cfg.scale))

    module = importlib.import_module("x{}.model.{}".format(cfg.scale, cfg.model))

    net = module.Net(scale=cfg.scale, group=cfg.group)

    state_dict = torch.load(cfg.ckpt_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    device = torch.device("cpu")
    net = net.to(device)

    ###########################################################
    # from thop import profile
    # from thop import clever_format
    # input = torch.randn(1, 3, 128, 128)
    # flops, params = profile(net, inputs=(input,))
    # print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    # flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    # print("params:", params)
    # scale 2
    # flops(G): 44.459
    # params: 626.328K
    # scale 3
    # flops(G): 85.083
    # params: 810.968K
    # scale 4
    # flops(G): 152.592
    # params: 774.040K

    input_shape = (1, 3, 128, 128)
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(net, input_data).eval()
    scripted_model.save(cfg.ckpt_path.replace(".pth", ".torchscript.pt"))
    scripted_model = torch.jit.load(cfg.ckpt_path.replace(".pth", ".torchscript.pt"))
    
    # # opset_version=10 RuntimeError: ONNX export failed: Couldn't export operator aten::pixel_shuffle
    # import onnx
    # torch.onnx.export(net, input_data, cfg.ckpt_path.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=13,
    # dynamic_axes= {
    #     "input": {2 : 'height', 3: 'width'},
    #     "output": {2: 'height', 3:'width'}}
    # )
    # shape_dict = {"input": input_shape}
    # onnx_model = onnx.load(cfg.ckpt_path.replace(".pth", ".onnx"))
    # ##########################################################


if __name__ == "__main__":
    cfg = parse_args()
    export(cfg)

