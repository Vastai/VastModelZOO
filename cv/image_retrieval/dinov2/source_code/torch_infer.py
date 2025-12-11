# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import sys
import pickle
import argparse

import torch
from torch import nn
import numpy as np
# import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
from PIL import Image, ImageFile

import hashlib

import sys
import utils


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
    # block_size_hacked = 1 * 8 * 8
    # c_group_hacked = 1
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

@torch.no_grad()
def extract_features(model, data_loader, use_cuda=False, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = []
    
    for samples, index in metric_logger.log_every(data_loader, 10):
        if use_cuda:
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            if args.infer_type == 'torch':
                feats = model(samples).clone()
            elif args.infer_type == 'onnx':
                ort_inputs = {model.get_inputs()[0].name: np.array(samples)}
                feats = model.run(None, ort_inputs)[0]
                feats = torch.from_numpy(feats)
            else:
                name = model.set_batch_size(1)
                input_image = get_activation_aligned_faster(np.array(samples).astype("float16"))
                model.set_input(name, 'input', 0, tvm.nd.array(input_image))
                model.run(name)
                feats = model.get_output(name, 0, 0).asnumpy()
                feats = torch.from_numpy(feats)

        features.append(feats)

    features = torch.stack(features, dim=1)
    features = torch.squeeze(features)

    return features

class OxfordParisDataset(torch.utils.data.Dataset):
    def __init__(self, dir_main, dataset, split, transform=None, imsize=None):
        if dataset not in ['roxford5k', 'rparis6k']:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        
        # cfg['imlist'] = cfg['imlist'][:10]
        # cfg['qimlist'] = cfg['qimlist'][:10]
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'
        cfg['dir_data'] = os.path.join(dir_main, dataset)
        cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')
        cfg['n'] = len(cfg['imlist'])
        cfg['nq'] = len(cfg['qimlist'])
        cfg['im_fname'] = config_imname
        cfg['qim_fname'] = config_qimname
        cfg['dataset'] = dataset
        self.cfg = cfg

        self.samples = cfg["qimlist"] if split == "query" else cfg["imlist"]
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = os.path.join(self.cfg["dir_images"], self.samples[index] + ".jpg")
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.imsize is not None:
            img.thumbnail((self.imsize, self.imsize), Image.LANCZOS)
        if self.transform is not None:
            img = self.transform(img)
        return img, index


def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])


def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])


def _make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4] + '_reg4'
    return f"dinov2_{compact_arch_name}{patch_size}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Retrieval on revisited Paris and Oxford')
    parser.add_argument('--pretrained_weights', default='/path/to/official_dinov2_fp16/dinov2_vitl14_reg4.onnx', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--data_path', default='/path/to/image_retrieval/', type=str)
    parser.add_argument('--dataset', default='roxford5k', type=str, choices=['roxford5k', 'rparis6k'])
    parser.add_argument('--infer_type', default='onnx', type=str, choices=['torch', 'onnx'])
    parser.add_argument('--multiscale', default=False, type=utils.bool_flag)
    parser.add_argument('--imsize', default=224, type=int, help='Image size')
    parser.add_argument('--use_cuda', default=False, type=utils.bool_flag)
    parser.add_argument('--arch', default='vit_large', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=14, type=int, help='Patch resolution of the model.')

    args = parser.parse_args()

    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize([args.imsize, args.imsize]),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = OxfordParisDataset(args.data_path, args.dataset, split="train", transform=transform, imsize=args.imsize)
    dataset_query = OxfordParisDataset(args.data_path, args.dataset, split="query", transform=transform, imsize=args.imsize)
    sampler = torch.utils.data.SequentialSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )
    print(f"train: {len(dataset_train)} imgs / query: {len(dataset_query)} imgs")

    if args.infer_type == 'torch':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        import onnxruntime as rt
        model = rt.InferenceSession(args.pretrained_weights, providers=["CPUExecutionProvider"])


    ############################################################################
    # Step 1: extract features
    train_features = extract_features(model, data_loader_train, args.use_cuda, multiscale=args.multiscale)
    query_features = extract_features(model, data_loader_query, args.use_cuda, multiscale=args.multiscale)

    if utils.get_rank() == 0:  # only rank 0 will work from now on
        # normalize features
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        query_features = nn.functional.normalize(query_features, dim=1, p=2)

        ############################################################################
        # Step 2: similarity
        sim = torch.mm(train_features.float(), query_features.T.float())
        ranks = torch.argsort(-sim, dim=0).cpu().numpy()

        ############################################################################
        # Step 3: evaluate
        gnd = dataset_train.cfg['gnd']
        # evaluate ranks
        ks = [1, 5, 10]
        # search for easy & hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = utils.compute_map(ranks, gnd_t, ks)
        # search for hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = utils.compute_map(ranks, gnd_t, ks)
        print('>> {}: mAP M: {}, H: {}'.format(args.dataset, np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        print('>> {}: mP@k{} M: {}, H: {}'.format(args.dataset, np.array(ks), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
    # dist.barrier()


'''
torch
dinov2_vitl14: dinov2_vitl14_pretrain.pth
>> roxford5k: mAP M: 74.91, H: 54.14
>> roxford5k: mP@k[ 1  5 10] M: [95.71 95.14 90.57], H: [91.43 73.71 63.11]

dinov2_vitl14_reg: dinov2_vitl14_reg4_pretrain.pth 
>> roxford5k: mAP M: 79.13, H: 59.21
>> roxford5k: mP@k[ 1  5 10] M: [97.14 95.14 91.57], H: [94.29 79.71 69.7 ]

onnx
dinov2_vitl14_reg: dinov2_vitl14_reg4.onnx 送测模型
>> roxford5k: mAP M: 79.41, H: 59.48
>> roxford5k: mP@k[ 1  5 10] M: [97.14 95.71 92.  ], H: [95.71 79.71 69.86]
'''