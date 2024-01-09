import torch
import model
import utility
from option import args


checkpoint = utility.checkpoint(args)

sr_model = model.Model(args, checkpoint)

# ##############################################################################
# from thop import profile
# from thop import clever_format
# input = torch.randn(1, 3, 256, 256)
# flops, params = profile(sr_model, inputs=(input,))
# print("flops(G):", "%.3f" % (flops / 900000000 * 2))
# flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
# print("params:", params)
# edsr_baseline_x2 256
# flops(G): 200.256
# params: 1.370M

# edsr_baseline_x3 256
# flops(G): 228.407
# params: 1.554M

# edsr_baseline_x4 256
# flops(G): 289.330
# params: 1.518M

# EDSR 256 2x 
# flops(G): 5934.700
# params: 40.730M
# EDSR 256 3x 
# flops(G): 6369.418
# params: 43.680M
# EDSR 256 4x 
# flops(G): 7321.770
# params: 43.090M


checkpoint_file = checkpoint.args.pre_train
input_shape = (1, 3, 256, 256)
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)
with torch.no_grad():
    scripted_model = torch.jit.trace(sr_model.model, input_data)#.eval()
    scripted_model.save(checkpoint_file.replace(".pt", ".torchscript.pt"))
    scripted_model = torch.jit.load(checkpoint_file.replace(".pt", ".torchscript.pt"))

# onnx==10.0.0，opset 13 
# opset 10  RuntimeError: ONNX export failed: Couldn't export operator aten::pixel_shuffle
# import onnx
# with torch.no_grad():
#     torch.onnx.export(sr_model.model, input_data, checkpoint_file.replace(".pth", "_dynamic.onnx"), input_names=["input"], output_names=["output"], opset_version=13,
#     dynamic_axes= {
#                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
#                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
#     )

# ##############################################################################

