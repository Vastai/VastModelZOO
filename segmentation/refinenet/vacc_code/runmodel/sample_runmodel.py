import os
import cv2
import glob
import torch
import hashlib
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F

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

def build_model(model_path, model_name):

    hash = hashlib.md5()
    hash.update(model_name.encode())
    md5_model = hash.hexdigest()
    model_key = f"{md5_model}:0:{model_name}"
    kwargs = {"name": model_key}

    ctx = tvm.vacc(0)
    loaded_json = open(os.path.join(model_path, model_name) + ".json").read()
    loaded_lib = tvm.module.load(os.path.join(model_path, model_name) + ".so")
    loaded_params = bytearray(open(os.path.join(model_path, model_name) + ".params", "rb").read())
    m = graph_runtime.create(loaded_json, loaded_lib, ctx, **kwargs)  # emu
    m.load_param(loaded_params)
    return m


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


if __name__ == '__main__':

    model_path = "deploy_weights/refinenet_lw_mobilenetv2-int8-kl_divergence-3_500_500-debug"
    model_name = "refinenet_lw_mobilenetv2"
    input_size = 500
    data_dir = "/home/simplew/dataset/seg/VOCdevkit/VOC2012/JPEGImages_val"
    gt_dir = "/home/simplew/dataset/seg/VOCdevkit/VOC2012/SegmentationClass"

    result_dir = "./vacc_result"
    os.makedirs(result_dir, exist_ok=True)


    transform = transforms.Compose([
        transforms.ToTensor(), #->[0,1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ->[-1,1]
    ])
    
    model = build_model(model_path, model_name)
    name =model.set_batch_size(1)

    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../..')
    from source_code.drsleep.score import SegmentationMetric

    metric = SegmentationMetric(21)
    
    image_files = glob.glob(os.path.join(data_dir, "*.jpg"))

    for image_path in tqdm(image_files):
        ori_image = Image.open(image_path)
        resize_image = ori_image.resize((input_size, input_size), Image.NEAREST).convert('RGB')
        images = transform(resize_image).unsqueeze(0).numpy()
        input_image = get_activation_aligned_faster(images.astype("float16"))
        model.set_input(name, 'input', 0, tvm.nd.array(input_image))
        model.run(name, 500)

        heatmap = model.get_output(name, 0, 0).asnumpy().astype("float32")
        
        # draw
        tvm_predict = torch.from_numpy(heatmap)
        tvm_predict = F.interpolate(tvm_predict, (input_size, input_size), mode='bilinear')     

        predict = tvm_predict[0].cpu().numpy()
        predict = np.asarray(predict, dtype="float32").transpose(1, 2, 0)

        predict_mask = cv2.resize(predict, ori_image.size, interpolation=cv2.INTER_CUBIC)

        colors = np.loadtxt("segmentation/refinenet/source_code/drsleep/voc2012_colors.txt").astype('uint8')
        color = colorize(predict_mask.argmax(axis=2).astype(np.uint8), colors)
        color.save(os.path.join(result_dir, os.path.basename(image_path)+".png"))
        
        ########################################################################################################
        # eval
        label_path = os.path.join(gt_dir, os.path.basename(image_path).replace(".jpg", ".png"))
        if not os.path.exists(label_path):
            continue
        gt = Image.open(label_path)
        gt = gt.resize(size=(input_size, input_size))
        target = np.array(gt).astype('int32')
        target[target == 255] = -1
        target = torch.from_numpy(target).long()

        metric.update(tvm_predict, target)
        pixAcc, mIoU = metric.get()
        print("{:s}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(image_path, pixAcc * 100, mIoU * 100))
        ########################################################################################################

"""

refinenet_resnet101-fp16-none-3_500_500-debug
validation pixAcc: 70.909, mIoU: 3.401
refinenet_resnet101-int8-kl_divergence-3_500_500-debug
validation pixAcc: 94.760, mIoU: 76.935


refinenet_lw_resnet50-fp16-none-3_500_500-debug
validation pixAcc: 24.097, mIoU: 1.381
refinenet_lw_resnet50-int8-kl_divergence-3_500_500-debug
validation pixAcc: 95.545, mIoU: 80.144


refinenet_lw_resnet101-fp16-none-3_500_500-debug
validation pixAcc: 70.083, mIoU: 3.384
refinenet_lw_resnet101-int8-kl_divergence-3_500_500-debug
validation pixAcc: 94.808, mIoU: 77.188

refinenet_lw_resnet152-fp16-none-3_500_500-debug
validation pixAcc: 10.812, mIoU: 0.693
refinenet_lw_resnet152-int8-kl_divergence-3_500_500-debug
validation pixAcc: 95.205, mIoU: 79.090

refinenet_lw_mobilenetv2-int8-kl_divergence-3_500_500-debug
validation pixAcc: 93.939, mIoU: 72.840

fp16 build error

03/20/2023 16:40:31 - vamc - ERROR    -   File "insert_dma.cc", line 381
TVMError: Check failed: sep_begin_ops.size() <= 2 (3 vs. 2) : The SEP OP for pipeline is expected at most 2 inputs, but given the number of input is 3
03/20/2023 16:40:31 - vamc - ERROR    - err info: 
Traceback (most recent call last):
  File "/home/simplew/code/080101_latest/0214_1.3.0/vaststream-1.3.0-rc2/tvm/python/tvm/_ffi/_ctypes/function.py", line 207, in __call__
    raise get_last_ffi_error()
tvm._ffi.base.TVMError:   File "insert_dma.cc", line 381
TVMError: Check failed: sep_begin_ops.size() <= 2 (3 vs. 2) : The SEP OP for pipeline is expected at most 2 inputs, but given the number of input is 3


"""