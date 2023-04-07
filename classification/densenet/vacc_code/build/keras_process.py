# ==============================================================================
# 
# Copyright (C) 2023 VastaiTech Technologies Inc.  All rights reserved.
# 
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :     vastml
@Time  : 2023/04/07 22:05:38
'''

import numpy as np
from PIL import Image


def get_image_data(image_file, input_shape = [1, 3, 224, 224]):
    """fix shape resize"""
    size = input_shape[2:]
    hints = [256, 256]
    mean = []
    std = []
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
    if len(mean) == 0 and len(std) == 0:
        # keras_model
        x = np.array(image)[np.newaxis, :].astype("float32")
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        return x.transpose((0, 3, 1, 2))
    elif mean[0] < 1 and std[0] < 1:
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


if __name__ == '__main__':
    img = get_image_data("../data/test/cls/car.jpg")
    print(img.shape)
