import torch
import onnx
import numpy as np
from unet import UNet


if __name__ == '__main__':

    checkpoint = 'ckpts/gaussian/n2n-gaussian.pt'

    model = UNet(in_channels=3)

    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    device = torch.device('cpu')
    model.eval()
    ##############################################################################

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

    # # # ckpts/gaussian/n2n-gaussian.pt (1, 3, 256, 256)
    # flops(G): 41.447
    # params: 700.611K
    

    input_shape = (1, 3, 256, 256)
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, input_data)#.eval()
        scripted_model.save(checkpoint.replace(".pt", ".torchscript.pt"))

    # onnx==10.0.0，opset 10
    import onnx
    with torch.no_grad():
        torch.onnx.export(model, input_data, checkpoint.replace(".pt", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11,
        # dynamic_axes= {
        #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
        #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
        dynamic_axes= {
                    "input": {0: 'batch_size'},
                    "output": {0: 'batch_size'}}
        )
