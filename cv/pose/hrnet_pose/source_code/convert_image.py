# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import  os
import numpy as np
import glob
from PIL import Image
import cv2



### 原图和flip图转换后的npz保存在同一个文件夹
### 命名示例： 000000000139.npz,000000000139_flip.npz
savepath = "./coco_flip_npz"
target_path = "./Documents/project/det_data/coco/val2017/*"

## vamp datalist.txt
datalist_txt = "./vamp_datalist.txt"

fin = open(datalist_txt, 'w')
img_list = glob.glob(target_path)
for file in img_list:
    img_data = cv2.imread(file)

    
    img_name = os.path.basename(file).split('.')[0]
    
    img = cv2.imread(file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    source_data = np.array(img)
    source_data = source_data.transpose(2,0,1)
    outnpz_1 = {}
    outnpz_1["input_0"] = source_data
    np.savez(os.path.join(savepath, img_name), **outnpz_1)
    fin.write(f"{os.path.join(savepath, img_name)}.npz\n")

    out = cv2.flip(img, 1)
    # out = img.transpose(Image.FLIP_LEFT_RIGHT)
    flip_data = np.array(out)
    flip_data = flip_data.transpose(2,0,1)
    outnpz_2 = {}
    outnpz_2["input_0"] = flip_data
    np.savez(os.path.join(savepath, img_name+'_flip'), **outnpz_2)
    fin.write(f"{os.path.join(savepath, img_name+'_flip')}.npz\n")
fin.close()
