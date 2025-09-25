import os
import cv2
import glob
import hashlib
import numpy as np
from PIL import Image

import tvm
from tvm.contrib import graph_runtime
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from classification.common.utils import get_activation_aligned_faster, find_file_by_suffix


def get_image_data(image_file, hints=[],input_shape = [1, 3, 256, 256]):
    """fix shape resize"""
    size = [input_shape[2],input_shape[3]]
    mean = [
        0.485,
        0.456,
        0.406
    ]
    std = [
        0.229,
        0.224,
        0.225
    ]

    image = Image.open(image_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if len(hints) != 0:
        y1 = max(0, int(round((hints[0] - size[0]) / 2.0)))
        x1 = max(0, int(round((hints[1] - size[1]) / 2.0)))
        y2 = min(hints[0], y1 + size[0])
        x2 = min(hints[1], x1 + size[1])
        image = image.resize(hints)
        image = image.crop((x1, y1, x2, y2))
    else:
        image = image.resize((size[1], size[0]))
    image = np.ascontiguousarray(image)
    if mean[0] < 1 and std[0] < 1:
        image = image.astype(np.float32)
        image /= 255.0
        image -= np.array(mean)
        image /= np.array(std)
    else:
        image = image - np.array(mean)  # mean
        image /= np.array(std)  # std
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

class Runner(object):
    def __init__(self,model_name, save_dir,vacc_weights_dir, model_input_name, model_input_shape):
        self.model_name = model_name
        self.save_dir = save_dir
        self.weights_dir = vacc_weights_dir
        self.m = self._get_model()
        self.model_input_name = model_input_name
        self.model_input_shape = model_input_shape

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _get_model(self):
        hash = hashlib.md5()
        hash.update(self.model_name.encode())
        md5_model = hash.hexdigest()
        model_key = f"{md5_model}::{self.model_name}"
        kwargs =  {"name": model_key}

        # 指定die
        ctx = tvm.vacc(0)
        with open(f"{self.weights_dir}/{self.model_name}.json") as f:
            loaded_json = f.read()
        loaded_lib = tvm.module.load(f"{self.weights_dir}/{self.model_name}.so")
        with open(f"{self.weights_dir}/{self.model_name}.params", "rb") as f:
            loaded_params = bytearray(f.read())
        m = graph_runtime.create(loaded_json, loaded_lib, ctx, **kwargs)  # emu
        m.load_param(loaded_params)
        return m

    def run(self, files_path,label_txt,save_img=False):
        files, _ = find_file_by_suffix(files_path, suffix_list=[".JPEG"], recursive=True)
        fin = open(os.path.join(self.save_dir, args.save_result_txt), "w")

        for file in tqdm(files):
            img = get_image_data(file, input_shape=self.model_input_shape)
            input_image = get_activation_aligned_faster(img.astype("float16"))
            name = self.m.set_batch_size(1)
            self.m.set_input(name, args.model_input_name, 0, tvm.nd.array(input_image))
            # predict
            self.m.run(name)
            output = self.m.get_output(name,0).asnumpy()
            # save results
            predictions = np.squeeze(output)
            predictions_new = np.zeros(1000)
            for i in range(1000):
                predictions_new[i] = predictions[i]
            top_k = predictions_new.argsort()[-(5) :][::-1]
            with open(label_txt) as f:
                classes_list = [cls.strip() for cls in f.readlines()]

            file_name = os.path.split(file)[-1]

            id = top_k[0]
            prob = "%.2f" % (predictions_new[id])
            name =  classes_list[id]
            text = f"id: {id},name: {name},prob: {prob}"
            for k, cls_id in enumerate(top_k):
                fin.write(
                    file
                    + ": "
                    + "Relay top-{} id: {}, prob: {:.8f}, class name: {}".format(
                        k, cls_id, predictions_new[cls_id], classes_list[cls_id]
                    )
                    + "\n"
                )
                # print(
                #     file
                #     + ": "
                #     + "Relay top-{} id: {}, prob: {:.8f}, class name: {}".format(
                #         k, cls_id, predictions_new[cls_id], classes_list[cls_id]
                #     )
                #     + "\n"
                # )
            if save_img:
                simage = cv2.imread(file)
                cv2.putText(simage, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imwrite(os.path.join(self.save_dir,file_name), simage)
        fin.close()

import argparse

def parse_int_list(value):
    return [int(x) for x in value.split(',')]

parser = argparse.ArgumentParser(description="RUN RESNET WITH VSX")
parser.add_argument("--file_path", type=str, default="/path/to/ILSVRC2012_img_val/", help="img dir path")
parser.add_argument("--model_weight_path", type=str, default="deploy_weights/ppcls_dlanet_run_model_fp16/", help="model info")
parser.add_argument("--model_name", type=str, default="mod", help="model info")
parser.add_argument("--model_input_name", type=str, default="input", help="model info")
parser.add_argument("--model_input_shape", type = parse_int_list, default = [1,3, 224, 224], help = "model input shape")
parser.add_argument("--label_txt", type=str, default="/path/to/ILSVRC2012_img_val/imagenet.txt", help="label txt")
parser.add_argument("--save_dir", type=str, default="./output", help="save_dir")
parser.add_argument("--save_result_txt", type=str, default="mod.txt", help="save_dir")

args = parser.parse_args()

if __name__ == "__main__":
    runner = Runner(model_name = args.model_name, save_dir= args.save_dir, 
                    vacc_weights_dir = args.model_weight_path, model_input_name = args.model_input_name, model_input_shape= args.model_input_shape)
    runner.run(files_path = args.file_path, label_txt = args.label_txt)

"""
ppcls dlanet fp16精度
top1_rate: 75.04 top5_rate: 92.396

timm dlanet  fp16精度
top1_rate: 75.03 top5_rate: 92.402

ucbdrive dlanet fp16精度
top1_rate: 73.028 top5_rate: 91.176
"""