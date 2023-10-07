"""
Inference ONNX model of MODNet

Arguments:
    --image-path: path of the input image (a file)
    --output-path: path for saving the predicted alpha matte (a file)
    --model-path: path of the ONNX model

Example:
python inference_onnx.py \
    --image-path=demo.jpg --output-path=matte.png --model-path=modnet.onnx
"""

import os
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import onnx
import torch
import onnxruntime
from metrics import metrics_class_dict, StreamSegMetrics


def get_scale_factor0(im_h, im_w, ref_size):
    # Get x_scale_factor & y_scale_factor to resize image
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h

    return x_scale_factor, y_scale_factor

def get_scale_factor(im_h, im_w, ref_size):
    # Get x_scale_factor & y_scale_factor to resize image

    im_rh = ref_size[0]
    im_rw = ref_size[1]

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h

    return x_scale_factor, y_scale_factor


def onnx_inference0(session, image_path):
    ref_size = 512
    ##############################################
    #  Main Inference part
    ##############################################

    # read image
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # normalize values to scale it between -1 to 1
    im = (im - 127.5) / 127.5   

    im_h, im_w, im_c = im.shape
    x, y = get_scale_factor(im_h, im_w, ref_size) 

    # resize image
    im = cv2.resize(im, None, fx = x, fy = y, interpolation = cv2.INTER_AREA)

    # prepare input shape
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis = 0).astype('float32')

    # Initialize session and get prediction
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})

    # refine matte
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation = cv2.INTER_AREA)

    return matte


def onnx_inference(session, image_path):
    ref_size = [480, 288]
    ##############################################
    #  Main Inference part
    ##############################################

    # read image
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # normalize values to scale it between -1 to 1
    im = (im - 127.5) / 127.5   

    im_h, im_w, im_c = im.shape
    x, y = get_scale_factor(im_h, im_w, ref_size) 

    # resize image
    # im = cv2.resize(im, None, fx = x, fy = y, interpolation = cv2.INTER_AREA)
    im = cv2.resize(im, ref_size[::-1], interpolation = cv2.INTER_AREA)

    # prepare input shape
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis = 0).astype('float32')

    # Initialize session and get prediction
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})

    # refine matte
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    # matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation = cv2.INTER_AREA)

    return matte


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default="datasets/PPM-100/image", help='path of the input image (a file)')
    parser.add_argument('--matte_dir', type=str, default="datasets/PPM-100/matte", help='path of the input image (a file)')
    parser.add_argument('--output_dir', type=str, default="./results", help='paht for saving the predicted alpha matte (a file)')
    parser.add_argument('--model-path', type=str, default="pretrained/modnet_photographic_portrait_matting.onnx" ,help='path of the ONNX model')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # metrics
    metrics_ins = {}
    metrics_data = {}
    metrics = ['mse', 'mad', 'grad', 'conn']

    # add seg_metrics
    seg_metrics = StreamSegMetrics(2)

    for key in metrics:
        key = key.lower()
        metrics_ins[key] = metrics_class_dict[key]()
        metrics_data[key] = None

    # define onnx model
    session = onnxruntime.InferenceSession(args.model_path, None)

    image_files = glob.glob(os.path.join(args.image_dir, "*"))
    for image_path in tqdm(image_files):
        # infer
        pred = onnx_inference(session, image_path)

        # eval
        gt = cv2.imread(os.path.join(args.matte_dir, os.path.basename(image_path)), 0)
        gt = cv2.resize(gt, pred.shape[::-1], interpolation=cv2.INTER_AREA)

        save_path = os.path.join(args.output_dir, os.path.basename(image_path))
        con = np.concatenate([gt, pred], axis=1)
        cv2.imwrite(save_path, con)

        for key in metrics_ins.keys():
            metrics_data[key] = metrics_ins[key].update(pred, gt, trimap=None)
        
        # add seg_metrics 
        gt[gt < 128] = 0
        gt[gt >= 128] = 1
        pred[pred < 128] = 0
        pred[pred >= 128] = 1
        seg_metrics.update(gt, pred)


    for key in metrics_ins.keys():
        metrics_data[key] = metrics_ins[key].evaluate()
    print("matting_metrics: \n", metrics_data)

    # add seg_metrics
    val_score = seg_metrics.get_results()
    print("seg_metrics: \n", seg_metrics.to_str(val_score))

"""
onnx

[512, 512]
{'mse': 0.004579846329783127, 'mad': 0.010063900518410147}

hw [480, 288]
{'mse': 0.010306540138649976, 'mad': 0.015813641839380502}
"""

"""
matting_metrics:

{'mse': 0.004500070830131148, 'mad': 0.011625731119152142, 'grad': 76.552794140625, 'conn': 102.82822099609375}

seg_metrics: 
 
Overall Acc: 0.987268
Mean Acc: 0.985027
FreqW Acc: 0.974931
Mean IoU: 0.969818
"""

"""
pretrained/modnet_photographic_portrait_matting.onnx

matting_metrics: 
 {'mse': 0.0094580874476697, 'mad': 0.014565194376361652, 'grad': 3.5139248394775398, 'conn': 1.8079100167846682}
seg_metrics: 
 
Overall Acc: 0.987484
Mean Acc: 0.985500
FreqW Acc: 0.975340
Mean IoU: 0.970898

"""