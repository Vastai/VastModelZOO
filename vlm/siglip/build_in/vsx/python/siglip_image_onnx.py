# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import numpy as np
import onnxruntime as ort   
from PIL import Image
import torch
import cv2
import torchvision.transforms as transforms
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        # print(f"new_unpad:{new_unpad}")
    
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        # print(f"dw, dh:{dw, dh}")
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        # print(f"dw, dh:{dw, dh}")
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        # print(f"top, bottom, left, right:{top, bottom, left, right}")
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)
class SiglipImageOnnx:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = self.load_onnx_model(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

    def load_onnx_model(self, onnx_path):
        sess_options = ort.SessionOptions()
        # 启用所有优化级别
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # session = ort.InferenceSession(onnx_path, sess_options)
        session = ort.InferenceSession(onnx_path)
        return session

    # def preprocess_image(self, image_path, input_shape=(224, 224)):
    #     """
    #     图像预处理：调整大小、归一化、转Tensor等
    #     input_shape: (H, W)
    #     """
    #     img = Image.open(image_path).convert('RGB')

    #     preprocess = transforms.Compose([
    #         transforms.Resize(input_shape),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])

    #     input_tensor = preprocess(img).unsqueeze(0).numpy()
    #     return input_tensor
    
    def preprocess_image(self, image, input_shape=(384, 384)):
        """
        图像预处理：调整大小、归一化、转Tensor等
        input_shape: (H, W)
        """
        # np.save('input_img.npy', image)
        img = letterbox(image, input_shape)[0]
        # np.save('letterbox.npy', img)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cpu')
        img = img.float()
        img /= 255.0
        if img.ndim == 3:
            img = img.unsqueeze(0)
        # np.save('before.npy', img.numpy())
        return img.numpy()

    def run_inference(self, session, input_tensor):
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        input_feed = dict(zip(input_names, input_tensor))
        # print(f"output_name:{output_names},, input_tensor:{input_tensor[0].shape} input_name:{input_feed}")
        outputs = session.run(output_names, input_feed)
        return outputs[0]
    
    def postprocess_output(self, output):
        """
        后处理：将输出结果进行解码、筛选等操作
        """
        # 假设输出是一个二维数组，每个元素是一个类别的得分
        # 这里简单地返回原始输出作为示例
        return output   
    
    def get_onnx_res_dict(self, path_onnx, data_input):     
        if isinstance(data_input[0], torch.Tensor):
            data_input = [e.cpu().numpy() for e in data_input]   
        
        input_names = [input.name for input in self.session.get_inputs()]
        output_names = [output.name for output in self.session.get_outputs()]

        input_feed = dict(zip(input_names, data_input))
        result = self.session.run(output_names, input_feed)
        return dict(zip(output_names, result))  
    
    def process(self, image):
        imgs = self.preprocess_image(image)
        imgs = imgs if isinstance(imgs, list) else [imgs]

        result = self.run_inference(self.session, imgs)
        return result
