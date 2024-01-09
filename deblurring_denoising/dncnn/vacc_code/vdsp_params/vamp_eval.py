import  os
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="MAKE DATA LIST")
    parse.add_argument("--gt_dir", type=str, default="/home/simplew/dataset/denoise/Set12/Set12", help="path to input source image folder")
    parse.add_argument("--input_npz_path", type=str, default="/home/simplew/dataset/denoise/Set12/datalist_npz.txt", help="vamp data list text file path")
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
            # output = np.clip(pred, 0, 1)
            heatmap = np.squeeze(heatmap)
            output = np.clip(heatmap, 0, 1)
            sr_img = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.draw_dir, file_name), sr_img)

            # load gt and compute psnr
            gt_path = os.path.join(args.gt_dir, file_name)
            image_gt = cv2.imread(gt_path, 0)
            image_gt = cv2.resize(image_gt, sr_img.shape[:2][::-1])
            
            psnr_value = compare_psnr(sr_img, image_gt)
            ssim_value = compare_ssim(sr_img, image_gt)
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)
            print("{}---> psnr: {}, ssim: {}".format(file_name, psnr_value, ssim_value))
    print("mean psnr: {}, mean ssim: {}".format(np.mean(psnr_list), np.mean(ssim_list)))


"""
dncnn-fp16-none-1_1_256_256-vacc
mean psnr: 29.68801101541599, mean ssim: 0.8465019391186898

dncnn-int8-percentile-1_1_256_256-vacc
mean psnr: 29.578934003489408, mean ssim: 0.8372848348435024
"""