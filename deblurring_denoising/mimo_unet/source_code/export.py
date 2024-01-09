import torch
import onnx
import numpy as np
from models.MIMOUNet import MIMOUNet, MIMOUNetPlus


if __name__ == '__main__':

    checkpoint = 'weights/MIMO-UNetPlus.pkl'

    # model = MIMOUNet()
    model = MIMOUNetPlus()

    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict['model'])
    device = torch.device('cpu')
    model.eval()
    ##############################################################################

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 360, 640)
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

    # # # MIMO-UNet (1, 3, 720, 1280)
    # flops(G): 2092.315
    # params: 6.807M

    # # # MIMO-UNet (1, 3, 360, 640)
    # flops(G): 523.079
    # params: 6.807M

    # # # MIMO-UNetPlus (1, 3, 360, 640)
    # flops(G): 1202.556
    # params: 16.108M
    
    input_shape = (1, 3, 720, 1280)
    input_shape = (1, 3, 360, 640)
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, input_data)#.eval()
        scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))

    # onnx==10.0.0，opset 10
    import onnx
    with torch.no_grad():
        torch.onnx.export(model, input_data, checkpoint.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
        # dynamic_axes= {
        #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
        dynamic_axes= {
                    "input": {0: 'batch_size'},
                    "output": {0: 'batch_size'}}
        )
    exit(0)
    ##############################################################################