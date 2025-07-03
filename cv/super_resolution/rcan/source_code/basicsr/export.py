import cv2
import torch
import numpy as np
from basicsr.archs.sr4k_arch import SR4K

if __name__ == '__main__':


    checkpoint = "./code/env/0427_sr4k/Training_Results/Models/Settings_4.pth"

    model = SR4K(3, 3)
    model_weights = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(model_weights["params"])
    model.eval()

    image_file = "datasets/DIV2K/DIV2K_valid_LR_bicubic/X2/0801x2.png"
    image = cv2.imread(image_file)
    img = cv2.resize(image, [1920, 1080]) # , interpolation=cv2.INTER_AREA
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) # HWC to CHW
    img = np.expand_dims(img, axis=0)
    img_t = torch.from_numpy(img).to("cpu")/255.
    output = model(img_t)

    draw = output.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
    img_np = np.clip(draw.detach().numpy(), 0, 1) * 255.0
    cv2.imwrite("draw.jpg", img_np)
    # ##############################################################################

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 1080, 1920)
    flops, params = profile(model, inputs=(input,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)

    # rcan basicsr (1, 3, 1080, 1920)
    # flops(G): 356.659
    # params: 297.699K

    input_shape = (1, 3, 1080, 1920) # nchw
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, input_data)#.eval()
        scripted_model.save(checkpoint.replace(".pth", "-1080.torchscript.pt"))
        scripted_model = torch.jit.load(checkpoint.replace(".pth", "-1080.torchscript.pt"))

    # onnx==10.0.0，opset 13
    # opset 10 can not export onnx in pixel_shuffle
    # import onnx
    # with torch.no_grad():
    #     torch.onnx.export(model, input_data, checkpoint.replace(".pth", "_dynamic.onnx"), input_names=["input"], output_names=["output"], opset_version=13,
    #     dynamic_axes= {
    #                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
    #                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
    #     )
    #     torch.onnx.export(model, input_data, checkpoint.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=13,
    #     # dynamic_axes= {
    #     #             "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
    #     #             "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
    #     )
    #     # ##############################################################################

