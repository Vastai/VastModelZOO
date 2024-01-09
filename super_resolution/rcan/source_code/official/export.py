import os
import torch

import model.rcan_modify as rcan

class Args:
    def __init__(self):
        self.n_resgroups = 3
        self.n_resblocks = 2
        self.n_feats     = 32
        self.reduction   = 16
        self.res_scale   = 1

        self.scale       = [2]
        self.rgb_range   = 255
        self.n_colors    = 3
        self.model       = 'rcan_modiy'
        self.test_only   = True

class Args_mini:
    def __init__(self):
        self.n_resgroups = 2
        self.n_resblocks = 2
        self.n_feats     = 16
        self.reduction   = 16
        self.res_scale   = 1

        self.scale       = [2]
        self.rgb_range   = 255
        self.n_colors    = 3
        self.model       = 'rcan_modiy'
        self.test_only   = True

device = torch.device('cpu')


checkpoint = "rcan.pth"
# checkpoint = "rcan2.pth"
args = Args()
# args = Args_mini()

model = rcan.make_model(args)

model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
model.eval()
model.to(device)


# ##############################################################################

# from thop import profile
# from thop import clever_format
# input = torch.randn(1, 3, 1080, 1920)
# flops, params = profile(model, inputs=(input,))
# print("flops(G):", "%.3f" % (flops / 900000000 * 2))
# flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
# print("params:", params)

# RCAN_MODIFY 1080_1920 3 2 32
# flops(G): 358.207
# params: 297.723K

# RCAN2 1080_1920 2 2 16
# flops(G): 80.331
# params: 63.547K

input_shape = (1, 3, 1080, 1920) # nchw
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)
with torch.no_grad():
    scripted_model = torch.jit.trace(model, input_data)#.eval()
    scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))
    scripted_model = torch.jit.load(checkpoint.replace(".pth", ".torchscript.pt"))

# onnx==10.0.0，opset 10 
# opset_version=10 can not export onnx in pixel_shuffle
# import onnx
with torch.no_grad():
    torch.onnx.export(model, input_data, checkpoint.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
    # dynamic_axes= {
    #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
    #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
    )

    # ##############################################################################


# import torch
# from tvm import relay

# pretrained_model = torch.jit.load("/home/simplew/code/080101_latest/0222_1.3.0_SR4k/code/model_best.torchscript.pt")
# mod, params = relay.frontend.from_pytorch(pretrained_model, [("input", [1, 3, 1080, 1920])])
# pass
