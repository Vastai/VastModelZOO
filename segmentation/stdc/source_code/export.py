import torch
from models.model_stages import BiSeNet

checkpoint = './checkpoints/STDC1-Seg/model_maxmIOU50.pth'

net = BiSeNet(
    backbone='STDCNet813',
    n_classes=19,
    use_boundary_2=False,
    use_boundary_4=False,
    use_boundary_8=True,
    use_boundary_16=False,
    use_conv_last=False)

net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
net.eval()

from thop import profile
from thop import clever_format
input = torch.randn(1, 3, 512, 512)

flops, params = profile(net, inputs=(input, ))
print("flops(G):", "%.3f" % (flops / 900000000 * 2))
flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
print("params:", params)

# model_maxmIOU50.pth
# flops(G): 52.089
# params: 14.242M

# model_maxmIOU75.pth
# flops(G): 52.089
# params: 14.242M
# 两个权重网络结构一致只是训练时的输入size缩放系数不一致

dummy_input = torch.randn(1, 3, 512, 512)
scripted_model = torch.jit.trace(net, dummy_input).eval()
scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))

torch.onnx.export(net, dummy_input, checkpoint.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
            dynamic_axes= {
                            "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
                            "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}
                            }
)