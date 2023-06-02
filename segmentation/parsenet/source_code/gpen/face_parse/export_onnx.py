import torch
import onnx
import os
import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '..')

from face_parse.face_parsing import FaceParse


# define model and load weights
faceparser = FaceParse(".", device="cpu")

model = faceparser.faceparse

# computer model flops and params
from thop import profile
from thop import clever_format
input = torch.randn(1, 3, 512, 512)
flops, params = profile(model, inputs=(input,))
print("flops(G):", "%.3f" % (flops / 900000000 * 2))
flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
print("params:", params)
# flops(G): 522.048
# params: 21.303M

# convert onnx and torchscript
checkpoint = "parsenet.pth"

input_shape = (1, 3, 512, 512)
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)

scripted_model = torch.jit.trace(model, input_data).eval()
scripted_model.save(checkpoint.replace(".pth", "-512.torchscript.pt"))
scripted_model = torch.jit.load(checkpoint.replace(".pth", "-512.torchscript.pt"))

# onnx==10.0.0，opset 10
torch.onnx.export(model, input_data, checkpoint.replace(".pth", "-512.onnx"), input_names=["input"], output_names=["output"], opset_version=10,
                # dynamic_axes= {
                #                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
                #                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}
                #                 }
)
shape_dict = {"input": input_shape}
onnx_model = onnx.load(checkpoint.replace(".pth", "-512.onnx"))