import os
import cv2
import glob
import onnx
import torch
import onnxruntime
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.nn.init as init
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr



class DnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


if __name__ == '__main__':

    # model = DnCNN()
    checkpoint = 'dncnn.pth'
    model = torch.load(checkpoint, map_location="cpu")
    model.eval() 

    ################################################################
    # from thop import profile
    # from thop import clever_format
    # input = torch.randn(1, 1, 256, 256)
    # flops, params = profile(model, inputs=(input,))
    # print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    # flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    # print("params:", params)

    # # dncnn.pth (1, 1, 256, 256)
    # # flops(G): 80.987
    # # params: 556.096K

    # input_shape = (1, 1, 256, 256)
    # shape_dict = [("input", input_shape)]
    # input_data = torch.randn(input_shape)
    # with torch.no_grad():
    #     scripted_model = torch.jit.trace(model, input_data)#.eval()
    #     scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))

    # # onnx==10.0.0，opset 10
    # import onnx
    # with torch.no_grad():
    #     torch.onnx.export(model, input_data, checkpoint.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=10,
    #     dynamic_axes= {
    #                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
    #                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
    #     )
    ################################################################
    hr_image_dir = "/home/simplew/dataset/denoise/Set12/Set12"
    lr_image_dir ="/home/simplew/dataset/denoise/Set12/Set12_noise"
    image_files = glob.glob(lr_image_dir + "/*.png")
    
    onnx_model = onnxruntime.InferenceSession('dncnn.onnx')

    psnr_list = []
    ssim_list = []
    for image_path in tqdm(image_files):
        image_src = cv2.imread(image_path, 0)
        image = cv2.resize(image_src, [256, 256])
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)

        # with torch.no_grad():
        #     heatmap = model(torch.from_numpy(image))
        #     output = heatmap.detach().numpy().astype(np.float32)
        
        # # onnx infer
        output = onnx_model.run([onnx_model.get_outputs()[0].name], {onnx_model.get_inputs()[0].name: image})[0]

        output = np.squeeze(output)
        output = np.clip(output, 0, 1)
        sr_img = (output * 255.0).round().astype(np.uint8)

        # eval
        image_gt = cv2.imread(os.path.join(hr_image_dir, os.path.basename(image_path)), 0)
        image_gt = cv2.resize(image_gt, output.shape[:2][::-1]) # , interpolation=cv2.INTER_AREA

        psnr_value = compare_psnr(image_gt, sr_img)
        ssim_value = compare_ssim(image_gt, sr_img)
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)

    psnr_avg = np.mean(psnr_list)
    ssim_avg = np.mean(ssim_list)
    print("mean psnr : {}, mean ssim : {}".format(psnr_avg, ssim_avg))


"""
deblurring_denoising/dncnn/vacc_code/vdsp_params/image2npz.py
Set12 256size
sigma = 25

dncnn.pth
mean psnr : 29.68849189845346, mean ssim : 0.8465395489855134

dncnn.onnx
mean psnr : 29.688491485651753, mean ssim : 0.8465395864681821
"""