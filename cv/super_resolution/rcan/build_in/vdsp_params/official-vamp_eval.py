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


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--gt_dir", type=str, default="./code/model_check/SR/BasicSR/datasets/DIV2K/DIV2K_valid_HR", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="./code/model_check/SR/sr/RCAN/npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="./code/vamc/vamp/0.2.0/outputs/rcan", help="vamp output folder")
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

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype("float32")
            
            # draw
            output = np.squeeze(heatmap)
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            cv2.imwrite(os.path.join(args.draw_dir, file_name), output)

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
Set5
deploy_weights/model_best-int8-max-3_512_512-vacc Set5
butterfly_LRBI_x2.png psnr: 28.35187015131426, ssim: 0.8833450219242097
woman_LRBI_x2.png psnr: 32.50871973733062, ssim: 0.9288326820927217
head_LRBI_x2.png psnr: 34.45756495334298, ssim: 0.8718652681702155
bird_LRBI_x2.png psnr: 34.4450995247809, ssim: 0.9195874798291457
baby_LRBI_x2.png psnr: 35.8353377234165, ssim: 0.9336015195330893
mean psnr: 33.11971841803705, mean ssim: 0.9336015195330893


DIV2K
model_best-int8-max-3_512_512-vacc
mean psnr: 28.708006322823664, mean ssim: 0.6997898408336339

model_best1080-int8-max-3_1080_1920-vacc
mean psnr: 32.65389316468518, mean ssim: 0.7661328269606028

"""
