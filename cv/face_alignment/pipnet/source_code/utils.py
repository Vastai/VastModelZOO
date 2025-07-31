# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch
import numpy as np
from scipy.integrate import simps

def post_process(pred, num_nb, net_stride,  input_size=256):
    outputs_cls = torch.Tensor(pred[0])
    outputs_x = torch.Tensor(pred[1])
    outputs_y = torch.Tensor(pred[2])
    outputs_nb_x = torch.Tensor(pred[3])
    outputs_nb_y = torch.Tensor(pred[4])

    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
    assert tmp_batch == 1

    outputs_cls = outputs_cls.view(tmp_batch*tmp_channel, -1)
    max_ids = torch.argmax(outputs_cls, 1)
    max_cls = torch.max(outputs_cls, 1)[0]
    max_ids = max_ids.view(-1, 1)
    max_ids_nb = max_ids.repeat(1, num_nb).view(-1, 1)

    outputs_x = outputs_x.view(tmp_batch*tmp_channel, -1)
    outputs_x_select = torch.gather(outputs_x, 1, max_ids)
    outputs_x_select = outputs_x_select.squeeze(1)
    outputs_y = outputs_y.view(tmp_batch*tmp_channel, -1)
    outputs_y_select = torch.gather(outputs_y, 1, max_ids)
    outputs_y_select = outputs_y_select.squeeze(1)

    outputs_nb_x = outputs_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)
    outputs_nb_x_select = outputs_nb_x_select.squeeze(1).view(-1, num_nb)
    outputs_nb_y = outputs_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
    outputs_nb_y_select = outputs_nb_y_select.squeeze(1).view(-1, num_nb)

    tmp_x = (max_ids%tmp_width).view(-1,1).float()+outputs_x_select.view(-1,1)
    tmp_y = (max_ids//tmp_width).view(-1,1).float()+outputs_y_select.view(-1,1)
    tmp_x /= 1.0 * input_size / net_stride
    tmp_y /= 1.0 * input_size / net_stride

    tmp_nb_x = (max_ids%tmp_width).view(-1,1).float()+outputs_nb_x_select
    tmp_nb_y = (max_ids//tmp_width).view(-1,1).float()+outputs_nb_y_select
    tmp_nb_x = tmp_nb_x.view(-1, num_nb)
    tmp_nb_y = tmp_nb_y.view(-1, num_nb)
    tmp_nb_x /= 1.0 * input_size / net_stride
    tmp_nb_y /= 1.0 * input_size / net_stride

    return tmp_x, tmp_y, tmp_nb_x, tmp_nb_y, outputs_cls, max_cls

def get_meanface(meanface_file, num_nb):
    with open(meanface_file) as f:
        meanface = f.readlines()[0]
        
    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i,:]
        dists = np.sum(np.power(pt-meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1+num_nb])
    
    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[],[]]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)
    
    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len
    
    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0]*10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1]*10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len

def compute_nme(lms_pred, lms_gt, norm):
    lms_pred = lms_pred.reshape((-1, 2))
    lms_gt = lms_gt.reshape((-1, 2))
    nme = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm 
    return nme

def compute_fr_and_auc(nmes, thres=0.1, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    return fr, auc

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
            labels_new[os.path.basename(image_name)] = target
        return labels_new
    labels_new = []
    for label in labels:
        image_name = label[0]
        target = label[1:]
        target = np.array([float(x) for x in target])
        labels_new.append([image_name, target])
    return labels_new