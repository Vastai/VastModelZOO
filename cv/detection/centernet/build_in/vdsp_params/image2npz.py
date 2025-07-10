import os
import cv2
import sys
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm

_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../..')

from source_code.CenterNet.src.lib.utils.image import get_affine_transform


def preprocess(img_file, scale=1.0):
    img_data = cv2.imread(img_file)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

    height, width = img_data.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)
    if True:
      inp_height, inp_width = 512, 512
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | 31) + 1
      inp_width = (new_width | 31) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(img_data, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    # inp_image = cv2.resize(image, (inp_width, inp_height))

    # inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1)# .reshape(1, 3, inp_height, inp_width)
    # images = torch.from_numpy(images)
    
    return images


def make_npz_text(args):
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path, exist_ok=True)

    with open(args.text_path, 'w') as f:
        files_list = glob.glob(args.dataset_path + "/*.jpg")

        for img_file in tqdm(files_list):

            # img_data = cv2.imread(img_file)
            # img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
            # data = np.array(img_data)
            # data = data.transpose(2,0,1) # hwc--> chw
            
            data = preprocess(img_file)

            data_dict = {"input_0": data}
            save_path = os.path.join(args.target_path, os.path.splitext(os.path.basename(img_file))[0]+ ".npz")
            np.savez(save_path, **data_dict)
            f.write(os.path.abspath(save_path) + "\n")


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="MAKE DATA LIST")
    parse.add_argument("--dataset_path", type=str, default="./code/eval/coco_val2017", help="path to input source image folder")
    parse.add_argument("--target_path", type=str, default="./code/modelzoo/vastmodelzoo/detection/centernet/coco_val2017_npz", help="path to output npz image folder")
    parse.add_argument("--text_path", type=str, default="datalist_npz.txt")
    args = parse.parse_args()
    print(args)

    make_npz_text(args)
