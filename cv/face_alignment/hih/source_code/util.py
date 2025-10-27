# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
# import copy
from scipy.integrate import simps
from math import ceil
from PIL import Image

def calc_nme(config,target_w_size,pred_heatmap, pred_offset=None):

    """
        Args: 
                target_w_size: tensor in pytorch (n,98,2)
                pred_heatmap : tensor in pytorch (n,98,64,64) 
            
        Return:
                Sum_ION : the sum Ion of this batch data
    """
    assert len(pred_heatmap.size()) == 4 , "the pred_heatmap must be 4 dim, use inference function"

    # decode_head_func = eval('decode_'+'hih'+'_head')
    preds = decode_hih_head(pred_heatmap,pred_offset)

    ION = []

    # target_w_size and preds : n, 98 , 2
    target_w_size *= config.heatmap_size
    target_np = target_w_size.reshape(pred_heatmap.size(0),-1,2)
    # target_np = target_w_size
    pred_np = preds.cpu().numpy().reshape(pred_heatmap.size(0),-1,2)

    for target,pred in zip(target_np,pred_np):
        diff = target - pred
        norm = np.linalg.norm(target[config.norm_indices[0]] - target[config.norm_indices[1]]) if config.norm_indices is not None else config.heatmap_size
        c_ION = np.sum(np.linalg.norm(diff,axis=1))/(diff.shape[0]*norm)
        ION.append(c_ION)


    Sum_ION = np.sum(ION) # the ion of this batch 
    # need div the dataset size to get nme

    return Sum_ION, ION

def compute_fr_and_auc(nmes, thres=0.10, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    nme = np.mean(nmes)

    print("NME %: {}".format(np.mean(nmes)*100))
    print("FR_{}% : {}".format(thres,fr*100))
    print("AUC_{}: {}".format(thres,auc))
    return nme, fr, auc

def get_label(label_path, ret_dict=False):
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [x.strip().split() for x in labels]
    if len(labels[0])==1:
        return labels
    if ret_dict:
        import os
        labels_new = {}
        for label in labels:
            image_name = label[0]
            target = label[1:]
            target = np.array([float(x) for x in target])
            labels_new[os.path.basename(image_name)] = target.reshape(-1, 2)
        return labels_new
    labels_new = []
    for label in labels:
        image_name = label[0]
        target = label[1:]
        target = np.array([float(x) for x in target])
        labels_new.append([image_name, target])
    return labels_new

def decode_hih_head(target_maps,offset_map):
    """
        target_maps : (n,98,64,64) float32
        offset_map : (n,98,4,4) float32

        return preds (n,98,2)
    """
    preds = decode_woo_head(target_maps).float()
    # offsets = decode_woo_head(offset_map) / torch.tensor([offset_map.size(3),offset_map.size(2)],dtype=torch.float32).cuda()
    offsets = decode_woo_head(offset_map) / torch.tensor([offset_map.size(3),offset_map.size(2)],dtype=torch.float32)
    preds.add_(offsets)
    
    return preds


def decode_woo_head(target_maps,offset_map=None):
    """
        Args:
            target_maps (n,98,64,64) tensor float32
            offset_map is None here.

        return : 
            preds (n,98,2)
    """
    max_v,idx = torch.max(target_maps.view(target_maps.size(0),target_maps.size(1),target_maps.size(2)*target_maps.size(3)), 2)
    preds = idx.view(idx.size(0),idx.size(1),1).repeat(1,1,2).float()
    max_v = max_v.view(idx.size(0),idx.size(1),1)
    pred_mask = max_v.gt(0).repeat(1, 1, 2).float()

    preds[..., 0].remainder_(target_maps.size(3))
    preds[..., 1].div_(target_maps.size(2)).floor_()

    preds.mul_(pred_mask)
    return preds

def pad_crop(image,target, cv2_flag=False):
    """
        add pad for the overflow points
        image need change data type
        border_pad : 8px
    """
    if cv2_flag:
        image_height = image.shape[0]
        image_width = image.shape[1]

    else:
        image_height, image_width = image.size
        

    l,t = np.min(target,axis=0)
    r,b = np.max(target,axis=0)
    
    # if the over border is left than grid_size, pass
    grid_size = 0.5 / image_height 
    
    if l > -grid_size and t > -grid_size and r < (1 + grid_size) and b < (1 + grid_size):
        target = np.maximum(target,0)
        target = np.minimum(target,1)
        return image,target
    border_pad_value = 8
    image_np = np.array(image).astype(np.uint8)
    border_size = np.zeros(4).astype('int') # upper bottom left right
    if l < 0:
        border_size[2] = ceil(-l * image_height) + border_pad_value #left
    if t < 0:
        border_size[0] = ceil(-t * image_width) + border_pad_value #upper
    if r > 1:
        border_size[3] = ceil((r-1) * image_height) + border_pad_value #right
    if b > 1:
        border_size[1] = ceil((b-1) * image_width) + border_pad_value #bottom
    border_img = np.zeros((image_width  + border_size[0] + border_size[1],
                           image_height + border_size[2] + border_size[3], 3)).astype(np.uint8)

    border_img[border_size[0] : border_size[0]+image_height, 
               border_size[2] : border_size[2]+image_width,:] = image_np
               
    image_pil = Image.fromarray(border_img.astype('uint8'), 'RGB')
    image_pil = image_pil.resize((image_height,image_width))
    target = (target * np.array([image_height,image_width]) + 
              np.array([border_size[2],border_size[0]])) /  np.array([border_img.shape[1],border_img.shape[0]])

    return image_pil, target


def concat_output(pred):
    pred0 = torch.Tensor(pred[0])
    pred1 = torch.Tensor(pred[1])
    # pred2 = torch.Tensor(pred[2])
    # pred3 = torch.Tensor(pred[3])
    # out_offset = torch.Tensor(pred[4])
    
    # output_pred = torch.stack((pred0, pred1, pred2, pred3), dim=1)
    # out_offset = pred4.unsqueeze(1)
    
    return pred0, pred1

def get_config(cfg_file, ):
    import yaml
    from pathlib import Path
    from easydict import EasyDict

    assert Path(cfg_file).exists(), "config file: {} is not exists".format(cfg_file)
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.safe_load(f)

    return EasyDict(config).config
    
