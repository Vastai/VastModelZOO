import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm

parse = argparse.ArgumentParser(description="MAKE DATA LIST")
parse.add_argument("--source_dir", type=str, default="testsets/Set12", help="path to input source image folder")
parse.add_argument("--noise_dir", type=str, default="testsets/Set12_noise", help="path to output noise image folder")
parse.add_argument("--target_dir", type=str, default="testsets/Set12_npz", help="path to output npz image folder")
parse.add_argument("--text_path", type=str, default="datalist_npz.txt")
args = parse.parse_args()
print(args)


def make_npz_text(args):
    os.makedirs(args.noise_dir, exist_ok=True)
    os.makedirs(args.target_dir, exist_ok=True)

    with open(args.text_path, 'w') as f:
        files_list = glob.glob(args.source_dir + "/*png")
        files_list.sort()
        for img_file in tqdm(files_list):

            image = cv2.imread(img_file, 0)
            image = image.astype(np.float32) / 255.

            np.random.seed(seed=0)  # for reproducibility
            noise_image = image + np.random.normal(0, 25 / 255.0, image.shape)
            noise_image = noise_image.astype(np.float32)
            noise_image = np.clip(noise_image, 0.0, 1.0)
            noise_image = (noise_image * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.noise_dir, os.path.basename(img_file)), noise_image)

            data = np.expand_dims(noise_image, -1)
            data = data.transpose(2,0,1) # hwc--> chw

            data_dict = {"input_0": data}
            save_path = os.path.join(args.target_dir, os.path.splitext(os.path.basename(img_file))[0]+ ".npz")
            np.savez(save_path, **data_dict)
            f.write(os.path.abspath(save_path) + "\n")


if __name__ == "__main__":
    make_npz_text(args)
