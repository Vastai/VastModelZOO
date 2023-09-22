import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image


def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,2]
    img[:,:,2] = ycbcr[:,:,1]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    img = np.array(img)
    return img

def PSNR(pred, gt, shave_border=0):
    import math
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--src_dir", type=str, default="/home/simplew/code/model_check/SR/sr/pytorch-vdsr/Set5_BMP/scale_4", help="hr gt image folder")
    parse.add_argument("--gt_dir", type=str, default="/home/simplew/code/model_check/SR/sr/pytorch-vdsr/Set5_BMP/hr", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="/home/simplew/code/vamc/vamp/0.2.0/outputs/drrn", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[256, 256], help="vamp input shape")
    parse.add_argument("--draw_dir", type=str, default="./npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    
    psnr_list = []
    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".bmp")

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)
            
            # draw
            tvm_predict = heatmap[0][0]
            output = np.clip(tvm_predict, 0, 1) * 255.0
            output = output.round().astype(np.uint8)

            im_b_ycbcr = cv2.imread(os.path.join(args.src_dir, file_name))
            im_b_ycbcr = cv2.cvtColor(im_b_ycbcr, cv2.COLOR_BGR2YCrCb)
            im_b_ycbcr = cv2.resize(im_b_ycbcr, args.input_shape)

            sr_img = colorize(output, im_b_ycbcr)
            cv2.imwrite(os.path.join(args.draw_dir, file_name), sr_img[:,:,::-1])

            # eval
            im_gt_ycbcr = cv2.imread(os.path.join(args.gt_dir, file_name.replace("_scale_4", "")))
            im_gt_ycbcr = cv2.cvtColor(im_gt_ycbcr, cv2.COLOR_BGR2YCrCb)
            im_gt_ycbcr = cv2.resize(im_gt_ycbcr, args.input_shape)

            psnr_vacc = PSNR(im_gt_ycbcr[:,:,0].astype(float), output, shave_border=4)
            psnr_list.append(psnr_vacc)
            print("{} psnr: {}".format(file_name, psnr_vacc))

        print("mean psnr: {}".format(np.mean(psnr_list)))

""" 
drrn-fp16-none-1_256_256-vacc
mean psnr: 30.940070623268888

drrn-int8-kl_divergence-1_256_256-vacc
mean psnr: 30.397305918789108
"""
