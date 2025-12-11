# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import argparse
import os
import math
import torch
import numpy as np
from concern.config import Configurable, Config
import onnxruntime, cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


import hashlib
import json

import numpy as np
import tvm
import vacc
from tvm.contrib import graph_runtime


def get_activation_aligned_faster(activation, dtype=np.float16, fc_mode=False, force_int8_layout_to_fp16=False):  # NCHW
    N = C = H = W = 1
    if len(activation.shape) == 2:
        N, C = activation.shape
        fc_mode = True
    elif len(activation.shape) == 5:
        N, C, H, W, B = activation.shape
    elif len(activation.shape) == 1:
        (C,) = activation.shape
    else:
        N, C, H, W = activation.shape
    h_group = w_group = c_group = 0
    if H == 1 and W == 1 and fc_mode == True:
        if dtype == np.float16:
            h_group, w_group, c_group = 1, 1, 256
        elif dtype == np.int8:
            h_group, w_group, c_group = 1, 1, 512
    else:
        if dtype == np.float16 or force_int8_layout_to_fp16:
            h_group, w_group, c_group = 8, 8, 4
        elif dtype == np.int8:
            h_group, w_group, c_group = 8, 8, 8
    pad_H, pad_W, pad_C = H, W, C
    pad_h, pad_w = 0, 0
    if H % h_group != 0:
        pad_h = h_group - H % h_group
        pad_H += pad_h
    if W % w_group != 0:
        pad_w = w_group - W % w_group
        pad_W += pad_w
    if C % c_group != 0:
        pad_c = c_group - C % c_group
        pad_C += pad_c
    # tensorize to WHC4c8h8w
    w_num = pad_W // w_group
    h_num = pad_H // h_group
    c_num = pad_C // c_group
    n_num = N
    block_size = w_group * h_group * c_group
    activation = activation.astype(dtype)
    assert(len(activation.shape) == 4)
    if (pad_h | pad_w) != 0:
        activation = np.pad(activation, ((0,0),(0,0),(0,pad_h),(0,pad_w)))
    np_arr = np.zeros((n_num, w_num, h_num, c_num, block_size), dtype)
    # for n in range(N):
    #     for c in range(C):
    #         for h in range(H):
    #             for w in range(W):
    #                 addr = (c % c_group) * h_group * w_group + (h % h_group) * w_group + (w % w_group)
    #                 if len(activation.shape) == 2:
    #                     np_arr[n, w // w_group, h // h_group, c // c_group, addr] = activation[n, c]
    #                 elif len(activation.shape) == 1:
    #                     np_arr[n, w // w_group, h // h_group, c // c_group, addr] = activation[n]
    #                 else:
    #                     np_arr[n, w // w_group, h // h_group, c // c_group, addr] = activation[n, c, h, w]
    block_size_hacked = 3 * 8 * 8
    c_group_hacked = 3
    for n in range(N):
        for c in range(c_num):
            c_index = c * c_group_hacked
            for h in range(h_num):
                h_index = h * h_group
                for w in range(w_num):
                    w_index = w * w_group
                    # print(activation[n, c_index:c_index+c_group_hacked, h_index:h_index+h_group, w_index:w_index+w_group].shape)
                    np_arr[n, w, h, c, :block_size_hacked] = activation[n, c_index:c_index+c_group_hacked, h_index:h_index+h_group, w_index:w_index+w_group].flatten()
    return np_arr



class RunFeature:
    def __init__(self):

        self.model_name = "ic15_resnet18_train"
        self.weights_dir = "./ic15_resnet18_train-vastml-onnx-int8-percentile-640_640"
        self.device = 0
        self.hw_config_path = "./VastDeploy/example/hw_config.json"
        self.input_name = "input"


        # 加载编译后模型
        self.m = self._get_model()

    def _get_model(self):
        with open(self.hw_config_path, "r") as f:
            hw_config_json = f.read()
        # kwargs = {"hwconfig_json_str": hw_config_json}
        hash = hashlib.md5()
        hash.update(self.model_name.encode())
        md5_model = hash.hexdigest()
        model_key = f"{md5_model}:0:{self.model_name}"
        kwargs = {"name": model_key, "hwconfig_json_str": hw_config_json}

        batch = int(json.loads(hw_config_json)["model"]["batch_size"])
        # 指定die
        ctx = tvm.vacc(self.device_id)
        loaded_json = open(f"{self.weights_dir}/{self.model_name}.json").read()
        loaded_lib = tvm.module.load(f"{self.weights_dir}/{self.model_name}.so")
        loaded_params = bytearray(open(f"{self.weights_dir}/{self.model_name}.params", "rb").read())
        m = graph_runtime.create(loaded_json, loaded_lib, ctx, **kwargs)  # emu
        m.load_param(loaded_params)
        assert (
            batch == 1
        ), f"The get feature model just support  batch is 1"
        return m

    def run(self,dataset):
        # 获取数据列表
        #dataset = get_image_data(img_path, self.input_size, self.input_hints, self.mean, self.std)
        input_image = get_activation_aligned_faster(dataset.astype("float16"))
        name = self.m.set_batch_size(1)

        self.m.set_input(name, self.input_name, 0, tvm.nd.array(input_image))
        # predict
        print("===============================begin infer==============================")
        self.m.run(name)
        # print("vacc outputs nums:", self.m.get_num_outputs())
        feature_map_list = []
        # get outputs
        #for output_index in range(self.m.get_num_outputs()):
        predictions = self.m.get_output(name, 0, 0).asnumpy().astype("float32")
        # print(img_path, " ", predictions.shape)
        # feature_map_list.append(predictions)
        return predictions


def main():
    parser = argparse.ArgumentParser(description='Convert model to ONNX')
    parser.add_argument('--exp', type=str, default='./DBNet/experiments/seg_detector/ic15_resnet18_deform_thre.yaml')
    parser.add_argument('--resume', type=str, default='./DBNet/workspace/SegDetectorModel-seg_detector/resnet18/L1BalanceCELoss/model/final', help='Resume from checkpoint')
    parser.add_argument('--output', type=str, default='./DBNet/ic15_resnet18_train.onnx', help='Output ONNX path')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--box_thresh', type=float, default=0.4,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--image_short_side', type=int, default=736,
                        help='The threshold to replace it in the representers')
    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    Demo(experiment, experiment_args, cmd=args).inference()


class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        self.structure = experiment.structure
        self.model_path = self.args['resume']
        self.output_path = self.args['output']

    def init_torch_tensor(self):
        # Use gpu or not
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

    def init_model(self):
        model = self.structure.builder.build(self.device_id)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        states = torch.load(path, map_location=self.device_id)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]

        # img = self.resize_image(img)
        img = cv2.resize(img, (640, 640))
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape

    def inference(self):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        model.eval()

        img = np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
        img = img.astype(np.float32)
        img = (img / 255. - 0.5) / 0.5  # torch style norm
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        dynamic_axes = {'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                        'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
        with torch.no_grad():
            img = img.to(self.device_id)
            torch.onnx.export(model.model.module, img, self.output_path, input_names=['input'],
                              output_names=['output'],  keep_initializers_as_inputs=False,
                              verbose=False, opset_version=10)# , dynamic_axes=dynamic_axes

        scripted_model = torch.jit.trace(model.model.module, img, strict=False)
        torch.jit.save(scripted_model, self.output_path.replace(".onnx", ".torchscript.pt"))

        session = onnxruntime.InferenceSession(self.output_path)
        # session = onnx.load(onnx_path)
        print("The model expects input shape: ", session.get_inputs()[0].shape)

        image_path = "/home/simplew/dataset/ocr/icdar2015/test_images/img_16.jpg"
        image_src = cv2.imread(image_path).astype('float32')

        IN_IMAGE_H = session.get_inputs()[0].shape[2]
        IN_IMAGE_W = session.get_inputs()[0].shape[3]

        # Input
        resized = cv2.resize(image_src, (640, 640), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in -= self.RGB_MEAN
        img_in /= 255.0
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        print("Shape of the network input: ", img_in.shape)

        batch = dict()
        batch['filename'] = [image_path]
        batch['shape'] = [image_src.shape[:2]]
        batch['image'], _ = self.load_image(image_path)

        # img_in = batch['image'].numpy()


        # Compute
        input_name = session.get_inputs()[0].name

        pred = session.run(None, {input_name: img_in})[0]
        print(pred)
        pred1 = torch.from_numpy(pred).float()
        output = self.structure.representer.represent(batch, pred1, is_output_polygon=False) 

        vis_image = self.structure.visualizer.demo_visualize(image_path, output)
        cv2.imwrite("./DBNet/demo_results/img_20000onnx.jpg", vis_image)


        ################VACC###############################
        run_func = RunFeature()
        feature_map = run_func.run(img_in)
        feature_map1 = np.expand_dims(feature_map, axis=0)
        feature_map1 = torch.from_numpy(feature_map1).float()
        output = self.structure.representer.represent(batch, feature_map1, is_output_polygon=False) 

        vis_image = self.structure.visualizer.demo_visualize(image_path, output)
        cv2.imwrite("./DBNet/demo_results/img_20000vacc.jpg", vis_image)
        ######################################


if __name__ == '__main__':
    main()
