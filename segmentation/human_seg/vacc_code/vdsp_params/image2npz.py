import os
import cv2
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm


def make_npz_text(args):
    os.makedirs(args.target_path, exist_ok=True)

    with open(args.text_path, 'w') as f:
        files_list = glob.glob(os.path.join(args.dataset_path, "ds*/img/*.png"))

        for img_file in tqdm(files_list):
            image_sub_name = img_file.split("/")[-3] + "/" + img_file.split("/")[-2] + "/" +  img_file.split("/")[-1]
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            npz_sub_name = img_file.split("/")[-3] + "/" + img_name + ".npz"

            img_data = cv2.imread(img_file)
            img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
            data = np.array(img_data)
            data = data.transpose(2,0,1) # hwc--> chw

            data_dict = {"input_0": data}

            save_path = os.path.join(args.target_path, npz_sub_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            np.savez(save_path, **data_dict)
            f.write(os.path.abspath(save_path) + "\n")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="MAKE DATA LIST")
    parse.add_argument("--dataset_path", type=str, default="/home/simplew/code/model_check/Human-Segmentation-PyTorch/dataset/Supervisely_Person_Dataset/src", help="path to input source image folder")
    parse.add_argument("--target_path", type=str, default="/home/simplew/code/model_check/Human-Segmentation-PyTorch/dataset/Supervisely_Person_Dataset/src_npz", help="path to output npz image folder")
    parse.add_argument("--text_path", type=str, default="npz_datalist.txt")
    args = parse.parse_args()
    print(args)

    make_npz_text(args)
