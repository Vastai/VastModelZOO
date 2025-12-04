# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import cv2
import time
import json
import glob
import math
import argparse
import threading
import numpy as np
from tqdm.contrib import tzip
from queue import Queue
from threading import Thread, Event
from typing import Dict, Generator, Iterable, List, Union

import vaststreamx as vsx


class VSXInference:
    def __init__(self,
        model_prefix_path: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        device_id: int = 0,
        batch_size: int = 1,
        is_async_infer: bool = False,
        model_output_op_name: str = "", ) -> None:


        if isinstance(vdsp_params_info, str):
            with open(vdsp_params_info) as f:
                vdsp_params_info_dict = json.load(f)
        
        self.model_size = vdsp_params_info_dict["OpConfig"]["OimageWidth"]
        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.preprocess_name = "preprocess_res"
        self.input_id = 0

        self.attr = vsx.AttrKey
        # self.device = vsx.set_device(self.device_id)
        assert vsx.set_device(self.device_id)==0
        # 构建model，模型三件套目录
        model_path = model_prefix_path
        self.model = vsx.Model(model_path, batch_size)
        # 输入预处理op
        # RGB_PLANAR
        self.fusion_op = vsx.Operator.load_ops_from_json_file(vdsp_params_info)[0]
        
        # 构建graph
        self.graph = vsx.Graph()
        self.model_op = vsx.ModelOperator(self.model)
        self.graph.add_operators(self.fusion_op, self.model_op)#

        # 构建stream
        self.infer_stream = vsx.Stream(self.graph, vsx.StreamBalanceMode.ONCE)
        if is_async_infer and len(self.model_output_op_name) > 0:
            self.infer_stream.register_operator_output(self.model_output_op_name, self.model_op)
        else:
            self.infer_stream.register_operator_output(self.model_op)
        
        # 预处理算子输出
        n,c,h,w = self.model.input_shape[0]
        self.infer_stream.register_operator_output(self.preprocess_name, self.fusion_op, [[(c,h,w), vsx.TypeFlag.FLOAT16]])

        self.infer_stream.build()
        
        self.current_id = -1
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}
        # 异步
        self.consumer = Thread(target=self.async_receive_infer)
        self.consumer.start()

    
    def async_receive_infer(self, ):
        while 1:
            try:
                result = None
                if len(self.model_output_op_name) > 0:
                    result = self.infer_stream.get_operator_output(self.model_output_op_name)
                else:
                    result = self.infer_stream.get_operator_output(self.model_op)
                if result is not None:
                    pre_process_tensor = self.infer_stream.get_operator_output(self.preprocess_name)
                    # 输出顺序和输入一致 
                    self.current_id += 1
                    input_id,height, width = self.input_dict[self.current_id]
                    model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in result[0]] ]
                    self.result_dict[input_id] = []
                    self.post_processing(input_id, height, width, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break
    
    def finish(self,):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def post_processing(self, input_id, height, width, stream_output_list):
        output_data = stream_output_list[0][0]
        self.result_dict[input_id] = output_data

    def _run(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            image = convert_rgb_to_ycbcr(image)
            image = np.stack([image[:,:,0], image[:,:,0], image[:,:,0]])
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)

        input_id = self.input_id
        self.input_dict[input_id] = (input_id, height, width)
        self.event_dict[input_id] = Event()


        self.infer_stream.run_async([input_image])
        
        self.input_id += 1

        return input_id

    def run(self, image:Union[str, np.ndarray]):
        input_id = self._run(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]

        return result

    def run_batch(self, images: Iterable[Union[str, np.ndarray]]) -> Generator[str, None, None]:

        queue = Queue(20)
        
        def input_thread():
            for image in images:
                input_id = self._run(image)
                queue.put(input_id)
            queue.put(None)

        thread = Thread(target=input_thread)
        thread.start()
        while True:
            input_id = queue.get()
            if input_id is None:
                break
            self.event_dict[input_id].wait()
            result = self.result_dict.pop(input_id)
            del self.event_dict[input_id]
            yield result
    
    def run_sync(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            image = convert_rgb_to_ycbcr(image)
            image = np.stack([image[:,:,0], image[:,:,0], image[:,:,0]])
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)

        output = self.infer_stream.run_sync([input_image])
        model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]] ]
        output_data = model_output_list[0][0]

        return output_data
        
def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])

def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


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


def set_config():
    parse = argparse.ArgumentParser(description="RUN WITH VSX")
    parse.add_argument(
        "--lr_image_dir",
        type=str,
        default="/path/to/DIV2K/DIV2K_valid_LR_bicubic/X2",
        help="lr_image_dir img dir",
    )
    parse.add_argument(
        "--hr_image_dir",
        type=str,
        default="/path/to/DIV2K/DIV2K_valid_HR",
        help="hr_image_dir img dir",
    )
    parse.add_argument(
        "--model_prefix_path",
        type=str,
        default="/path/to/model_plain_jit_new-int8-percentile-1_1_1080_1920-vacc/mod",
        help="model info")
    parse.add_argument(
        "--vdsp_params_info",
        type=str,
        default="../build_in/vdsp_params/official-ecbsr-vdsp_params.json",
        help="vdsp op info",
    )
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument("--batch", type=int, default=1, help="bacth size")
    parse.add_argument("--save_dir", type=str, default="./vsx_results", help="save result")
    args = parse.parse_args()
    print(args)

    return args


if __name__ == '__main__':
    args = set_config()

    vsx_inference = VSXInference(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )

    # Test multiple images
    image_files = glob.glob(os.path.join(args.lr_image_dir, "*.png"))

    os.makedirs(args.save_dir, exist_ok=True)

    results = vsx_inference.run_batch(image_files)
    #############################################################################
    # import onnxruntime
    # session = onnxruntime.InferenceSession("./code/model_check/SR/sr/ECBSR/model_ecbsr-1080_1920_sim.onnx")
    
    # import torch
    # device = torch.device('cpu')
    # model  = torch.jit.load("./code/model_check/SR/sr/ECBSR/experiments/ecbsr-x2-m4c8-prelu-2023-0714-1645/models/model_x2_940_plain.torchscript.pt", map_location=device)
    # model = model.to(device)
    # model.eval()
    #############################################################################

    psnr_list = []
    ssim_list = []
    # results = image_files
    for (image_path, result) in tzip(image_files, results):
        #############################################################################
        image_lr = cv2.imread(image_path)
        image_lr = cv2.resize(image_lr, (1920, 1080))
        image_lr = cv2.cvtColor(image_lr, cv2.COLOR_BGR2RGB)
        ycbcr = convert_rgb_to_ycbcr(image_lr.astype(np.float32))
        # note onnx and jit infer with shape: [1, 1, 1080, 1920]
        image = ycbcr[:,:,0]
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)

        ## for onnxruntime
        # input_name = session.get_inputs()[0].name
        # output_name = session.get_outputs()[0].name
        # heatmap = session.run([output_name], {input_name: image})[0]
        # result = np.squeeze(heatmap, axis=0)

        ## for torch.jit
        # with torch.no_grad():
        #     heatmap = model(torch.from_numpy(image))
        # result = np.squeeze(heatmap.detach().numpy(), axis=0)
        #############################################################################

        # post process
        ycbcr = cv2.resize(ycbcr, (0,0), fx=2,fy=2)
        # combine vacc output Y with source image cb and cy
        sr = np.array([result[0], ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        sr = np.clip(convert_ycbcr_to_rgb(sr), 0.0, 255.0)
        output = sr.astype(np.uint8)[:,:,::-1]
        cv2.imwrite(os.path.join(args.save_dir, os.path.basename(image_path)), output)
        
        # eval
        image_gt = cv2.imread(os.path.join(args.hr_image_dir, os.path.basename(image_path).replace("x2", "")))
        image_gt = cv2.resize(image_gt, output.shape[:2][::-1]) # , interpolation=cv2.INTER_AREA

        vacc_psnr = calculate_psnr(image_gt, output)
        vacc_ssim = calculate_ssim(image_gt, output)
        print("{} psnr: {}, ssim: {}".format(image_path, vacc_psnr, vacc_ssim))
        psnr_list.append(vacc_psnr)
        ssim_list.append(vacc_ssim)

    print("mean psnr: {}, mean ssim: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
    vsx_inference.finish()
    
"""
https://github.com/xindongzhang/ECBSR
此来源模型基于Ycbcr的Y通道数据进行训练, 所以原始模型推理shape为[1, 1, 1080, 1920]
但vacc的vdsp只能处理三通道数据, 所以vacc推理时叠加三个Y通道进行推理, 在结果中再把cb和cr通道叠加回去, 最后再转换行回rgb
流程复杂, 可能单通道的模型都不太适合我们的推理框架

# DIV2K_valid_LR_bicubic/X2/

model_ecbsr-1080_1920.torchscript.pt
mean psnr: 32.59298674375998, mean ssim: 0.7749856646062577

model_ecbsr_jit-fp16-none-1_1_1080_1920-vacc
mean psnr: 31.583973032954074, mean ssim: 0.750445889684897

model_ecbsr_jit-int8-percentile-1_1_1080_1920-vacc
mean psnr: 31.50376265871793, mean ssim: 0.744641008963285

model_plain-1080_1920.torchscript.pt
mean psnr: 32.59538342872409, mean ssim: 0.7748545753681958

model_plain_jit-fp16-none-1_1_1080_1920-vacc
mean psnr: 32.30710768597619, mean ssim: 0.7740473069736203

model_plain_jit-int8-percentile-1_1_1080_1920-vacc
mean psnr: 32.34440159638396, mean ssim: 0.7689059544343012
"""