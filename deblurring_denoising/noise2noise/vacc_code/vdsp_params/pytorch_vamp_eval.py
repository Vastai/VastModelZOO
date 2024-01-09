import  os
import cv2
import glob
import math
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_psnr(img1, img2):
    img1_ = img1.copy().astype(np.float64)
    img2_ = img2.copy().astype(np.float64)
    mse = np.mean((img1_-img2_)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0/math.sqrt(mse))


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

    parse = argparse.ArgumentParser(description="MAKE DATA LIST")
    parse.add_argument("--gt_dir", type=str, default="denoising/SIDD/val/target", help="path to input source image folder")
    parse.add_argument("--input_npz_path", type=str, default="idr_npz_datalist.txt", help="vamp data list text file path")
    parse.add_argument("--out_npz_dir", type=str, default="/home/simplew/code/vamc/vamp/2.1.0/outputs", help="path to vamp output npz image folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[256, 256], help="vamp input shape")
    parse.add_argument("--draw_dir", type=str, default="draw_npz_result", help="path to result image dolder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*"))
    output_npz_list.sort()

    psnr_list = []
    ssim_list = []
    with open(args.input_npz_path, 'r') as f:
        input_images = f.readlines()
        for index, line in enumerate(tqdm(input_images)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".png")
            
            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[index]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)
            
            # load out from vamp npz
            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)
            output = np.squeeze(heatmap)
            output = np.clip(output, 0, 1)
            output = np.clip(output*255, 0, 255)
            sr_img = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            cv2.imwrite(os.path.join(args.draw_dir, file_name), sr_img)

            # load gt and compute psnr
            gt_path = os.path.join(args.gt_dir, file_name)
            image_gt = cv2.imread(gt_path)
            image_gt = cv2.resize(image_gt, sr_img.shape[:2][::-1])
            
            psnr_value = calculate_psnr(sr_img, image_gt)
            ssim_value = calculate_ssim(sr_img, image_gt)
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)
            print("{}---> psnr: {}, ssim: {}".format(file_name, psnr_value, ssim_value))
    print("mean psnr: {}, mean ssim: {}".format(np.mean(psnr_list), np.mean(ssim_list)))


"""


"""