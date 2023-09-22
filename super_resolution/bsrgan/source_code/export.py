import os
import torch
from models.network_rrdbnet import RRDBNet as net


def main():

    # 'BSRNet' for scale factor 4
    # 'BSRGAN' for scale factor 4
    # 'BSRGANx2' for scale factor 2
    model_names = ['BSRNet', 'BSRGAN', 'BSRGANx2']    

    sf = 4
    device = torch.device('cpu')

    for model_name in model_names:
        if model_name in ['BSRGANx2']:
            sf = 2
        model_path = os.path.join('model_zoo', model_name+'.pth') # set model path

        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

        ###########################################################
        from thop import profile
        from thop import clever_format
        input = torch.randn(1, 3, 128, 128)
        flops, params = profile(model, inputs=(input,))
        print("flops(G):", "%.3f" % (flops / 900000000 * 2))
        flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
        print("params:", params)
        # # BSRNet
        # # flops(G): 652.697
        # # params: 16.698M
        
        # # BSRGAN
        # # flops(G): 652.697
        # # params: 16.698M

        # # BSRGANx2
        # # flops(G): 615.207
        # # params: 16.661M

        input_shape = (1, 3, 128, 128)
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        scripted_model.save(model_path.replace(".pth", ".torchscript.pt"))
        scripted_model = torch.jit.load(model_path.replace(".pth", ".torchscript.pt"))
        
        import onnx
        torch.onnx.export(model, input_data, model_path.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=10,
                        dynamic_axes= {
                                    "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
                                    "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}}
                        )
        onnx_model = onnx.load(model_path.replace(".pth", ".onnx"))
        # ##########################################################
if __name__ == '__main__':

    main()

"""

DIV2k 4x

torch BSRGAN x4
mean psnr: 23.023397616435542, mean ssim: 0.6361975193912127

torch BSRGANx2
mean psnr: 25.053282942106613, mean ssim: 0.6542593756079789

torch BSRNet x4
mean psnr: 24.532544491015205, mean ssim: 0.6754869013081449
"""