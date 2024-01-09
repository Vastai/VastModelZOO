#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os.path as osp
import os
import cv2
from transform import *
from PIL import Image
import shutil
import numpy as np

face_data = '/home/simplew/dataset/sr/CelebAMask-HQ/CelebA-HQ-img'
face_sep_mask = '/home/simplew/dataset/sr/CelebAMask-HQ/CelebAMask-HQ-mask-anno'

test_path = '/home/simplew/dataset/sr/CelebAMask-HQ/bisegnet_test_img'
mask_path = '/home/simplew/dataset/sr/CelebAMask-HQ/bisegnet_test_mask'
os.makedirs(test_path, exist_ok=True)
os.makedirs(mask_path, exist_ok=True)

counter = 0
total = 0
for i in range(15):

    atts = [
        'skin', 'l_brow', 'r_brow',
        'l_eye', 'r_eye', 'eye_g',
        'l_ear', 'r_ear', 'ear_r',
        'nose', 'mouth', 'u_lip',
        'l_lip', 'neck', 'neck_l',
        'cloth', 'hair', 'hat']

    ok_label_list = [
        'skin', 'nose', 'eye_g',
        'l_eye', 'r_eye', 'l_brow',
        'r_brow', 'l_ear', 'r_ear',
        'mouth', 'u_lip', 'l_lip',
        'hair', 'hat', 'ear_r',
        'neck_l', 'neck', 'cloth']

    for j in range(i * 2000, (i + 1) * 2000):

        mask = np.zeros((512, 512))

        for l, att in enumerate(atts, 1):
            total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = osp.join(face_sep_mask, str(i), file_name)

            if os.path.exists(path):
                counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))
                # print(np.unique(sep_mask))

                mask[sep_mask == 225] = l
        

        shutil.copy2('{}/{}.jpg'.format(face_data, j), '{}/{}.jpg'.format(test_path, j))
        cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)

        print(j)

print(counter, total)