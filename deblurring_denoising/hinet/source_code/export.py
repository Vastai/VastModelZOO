import os
import torch
from basicsr.models import create_model
from basicsr.train import parse_options


opt = parse_options(is_train=False)
net = create_model(opt)
model = net.net_g
model.eval()


# ##############################################################################

from thop import profile
from thop import clever_format
input = torch.randn(1, 3, 256, 256)
flops, params = profile(model, inputs=(input,))
print("flops(G):", "%.3f" % (flops / 900000000 * 2))
flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
print("params:", params)

# HINet-SIDD-0.5x.pth 256
# flops(G): 95.283
# params: 22.175M
# HINet-SIDD-1x.pth 256
# flops(G): 379.365
# params: 88.666M


checkpoint = opt['path']['pretrain_network_g']
input_shape = (1, 3, 256, 256) # nchw
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)
with torch.no_grad():
    scripted_model = torch.jit.trace(model, input_data)#.eval()
    scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))
    scripted_model = torch.jit.load(checkpoint.replace(".pth", ".torchscript.pt"))

# onnx==10.0.0，opset 10 
# can not export onnx in pixel_shuffle
import onnx
with torch.no_grad():
    torch.onnx.export(model, input_data, checkpoint.replace(".pth", "_dynamic.onnx"), input_names=["input"], output_names=["output"], opset_version=10,
    dynamic_axes= {
                "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
                "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
    )

    # ##############################################################################
