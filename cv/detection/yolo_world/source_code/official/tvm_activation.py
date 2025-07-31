# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import numpy as np


def get_activation_aligned_faster(activation, dtype=np.float16, fc_mode=False, force_int8_layout_to_fp16=False, is_input=True):  # NCHW
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
    if activation.dtype != dtype:
        activation = activation.astype(dtype)
    assert(len(activation.shape) == 4)
    if (pad_h | pad_w) != 0:
        activation = np.pad(activation, ((0,0),(0,0),(0,pad_h),(0,pad_w)))
    np_arr = np.zeros((n_num, w_num, h_num, c_num, block_size), dtype)
    block_size_hacked = 3 * 8 * 8 if is_input else 4 * 8 * 8
    c_group_hacked = 3 if is_input else 4
    for n in range(N):
        for c in range(c_num):
            c_index = c * c_group_hacked
            for h in range(h_num):
                h_index = h * h_group
                for w in range(w_num):
                    w_index = w * w_group
                    np_arr[n, w, h, c, :block_size_hacked] = activation[n, c_index:c_index+c_group_hacked, h_index:h_index+h_group, w_index:w_index+w_group].flatten()
    return np_arr


def bert_get_activation_fp16_A(activation, rep_dtype=None): # NCHW
    #pdb.set_trace()
    if activation.ndim == 2:
        M, K = activation.shape
        activation = activation.reshape((1,M,K))
    N, M, K = activation.shape

    m_group, k_group = 16, 16

    pad_M, pad_K = M, K
    if M % m_group != 0:
        pad_m = m_group - M % m_group
        pad_M += pad_m

    if K % k_group != 0:
        pad_k = k_group - K % k_group
        pad_K += pad_k

    # tensorize to MK16m16k
    n_num = N
    m_num = pad_M // m_group
    k_num = pad_K // k_group
    block_size = m_group * k_group
    activation = activation.astype(np.float16)
    np_arr = np.zeros((n_num, m_num, k_num, block_size), np.float16)

    for n in range(N):
        for m in range(M):
            for k in range(K):
                addr = (m % m_group) * k_group + (k % k_group)
                np_arr[n, m//m_group, k//k_group, addr] = activation[n, m, k]
    return np_arr


def gemm_align_z(data,shape = [128,768],core_num = 1):
    h,w = shape[0], shape[1]
    if w%(16*core_num):
        raise ValueError("w should be multiple of 16*core_num, but w = ",w)
    if h%16:
        raise ValueError("h should be multiple of 16, but h = ",h)

    data = np.reshape(data, newshape=[h,w])
    data2 = data.reshape([h//16,16,w//16,16])
    data2 = np.transpose(data2,[0,2,1,3])  # [h//16 w//16, 16,16]   align with Z

    data3 = data2.reshape([h//16,core_num, w//16//core_num, 16, 16])  # split with 8 core
    data3 = np.transpose(data3, [1,0,2,3,4])  # [core_num, h//16, w//16//core_num,16,16]
    return data3