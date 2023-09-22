import os
import math
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])

def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])



if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--gt_dir", type=str, default="/home/simplew/code/model_check/SR/BasicSR/datasets/DIV2K/DIV2K_valid_HR", help="hr gt image folder")
    parse.add_argument("--lr_dir", type=str, default="/home/simplew/code/model_check/SR/BasicSR/datasets/DIV2K/DIV2K_valid_LR_bicubic/X2", help="lr image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="/home/simplew/code/vamc/vamp/2.1.0/npz_output", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[1080, 1920], help="vamp input shape h,w")
    parse.add_argument("--draw_dir", type=str, default="./npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    
    psnr_list = []
    ssim_list = []
    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")
            # read source lr, get cb cr 
            image_lr = cv2.imread(os.path.join(args.lr_dir, file_name))
            image_lr = cv2.resize(image_lr, args.input_shape[::-1])
            image_lr = cv2.cvtColor(image_lr, cv2.COLOR_BGR2RGB)
            ycbcr = convert_rgb_to_ycbcr(image_lr.astype(np.float32))

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype("float32")
            heatmap = heatmap[0]
            
            # draw
            ycbcr = cv2.resize(ycbcr, (0,0), fx=2,fy=2)
            # combine vacc output Y with source image cb and cy
            sr = np.array([heatmap[0], ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            sr = np.clip(convert_ycbcr_to_rgb(sr), 0.0, 255.0)
            output = sr.astype(np.uint8)[:,:,::-1]
            # cv2.imwrite(os.path.join(args.draw_dir, file_name), output)

            # load gt and compute psnr
            image_gt = cv2.imread(os.path.join(args.gt_dir, os.path.basename(file_name).replace("x2", ""))) # set5 use ("LRBI", "HR") # div2k use ("x2", "")
            image_gt = cv2.resize(image_gt, output.shape[:2][::-1]) # , interpolation=cv2.INTER_AREA

            vacc_psnr = calculate_psnr(image_gt, output)
            vacc_ssim = calculate_ssim(image_gt, output)
            psnr_list.append(vacc_psnr)
            ssim_list.append(vacc_ssim)
            print("{} psnr: {}, ssim: {}".format(file_name, vacc_psnr, vacc_ssim))
        print("mean psnr: {}, mean ssim: {}".format(np.mean(psnr_list), np.mean(ssim_list)))


"""
https://github.com/xindongzhang/ECBSR
此来源模型基于Ycbcr的Y通道数据进行训练, 所以原始模型推理shape为[1, 1, 1080, 1920]
但vacc的vdsp只能处理三通道数据, 所以vacc推理时叠加三个Y通道进行推理, 在结果中再把cb和cr通道叠加回去, 最后再转换行回rgb
流程复杂, 可能单通道的模型都不太适合我们的推理框架

model_ecbsr_jit-int8-percentile-1_1_1080_1920-vacc
mean psnr: 31.503762658717925, mean ssim: 0.8618871242823097
"""
