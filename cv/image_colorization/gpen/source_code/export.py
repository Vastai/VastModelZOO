# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import cv2
import time
import torch
import numpy as np
import __init_paths
from face_model.face_gan import FaceGAN

if __name__ == '__main__':

    # 直接指定模型路径
    # https://github.com/yangxy/GPEN/blob/main/face_model/face_gan.py#L17C16-L17C16
    # self.mfile = "path/to/gpen.pth"
    facegan = FaceGAN(base_dir=".", in_size=512, out_size=512, model='GPEN', channel_multiplier=2, narrow=1,  device="cpu")
    
    # ###############################################################

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, facegan.in_resolution, facegan.in_resolution)
    flops, params = profile(facegan.model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)
    # flops(G): 0  not supported!
    # params: 71.01 M

    checkpoint = facegan.mfile

    input_shape = (1, 3, facegan.in_resolution, facegan.in_resolution)
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)

    scripted_model = torch.jit.trace(facegan.model, input_data).eval()
    scripted_model.save(checkpoint.replace(".pth", "-" + str(facegan.in_resolution) + ".torchscript.pt"))
    scripted_model = torch.jit.load(checkpoint.replace(".pth", "-" + str(facegan.in_resolution) + ".torchscript.pt"))

    # onnx==10.0.0，opset 10
    import onnx
    torch.onnx.export(facegan.model, input_data, checkpoint.replace(".pth", "-" + str(facegan.in_resolution) + ".onnx"), input_names=["input"], output_names=["output"], opset_version=11)
    shape_dict = {"input": input_shape}
    onnx_model = onnx.load(checkpoint.replace(".pth", "-" + str(facegan.in_resolution) + ".onnx"))


    import onnxruntime
    import numpy as np
    from torch.autograd import Variable

    onnx_file_name = checkpoint.replace(".pth", "-" + str(facegan.in_resolution) + ".onnx")
    onnxruntime_engine = onnxruntime.InferenceSession(onnx_file_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    dummy_input = Variable(torch.randn(1, 3, facegan.in_resolution, facegan.in_resolution))
    with torch.no_grad():
        torch_output = facegan.model(dummy_input)
    onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
    onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)
    np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-02, atol=1e-02)
    print("Example: Onnx model has been tested with ONNXRuntime, the result looks good !")

    # ###############################################################