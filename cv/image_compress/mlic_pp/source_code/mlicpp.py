# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from compressai.models import CompressionModel
from compressai.ops import ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils.func import update_registered_buffers, get_scale_table
from utils.ckbd import *
from modules.transform import *

import os
import numpy as np
from onnxsim import simplify
import onnx
from scipy import spatial
import onnxruntime as ort
import hashlib

flag_export_decompress = True
       
def get_hash(file_path):
    """Calculate the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# 通过 cos 相似度比较
def get_cosine(res_before, res_after, thresh_hold=1e-8):
    print('res_before: {f}'.format(f=res_before.shape))
    print('res_after: {f}'.format(f=res_after.shape))
    if res_after is not None and res_before is not None:
        res_before = res_before.flatten().astype("float32")
        res_after = res_after.flatten().astype("float32")
        cos_sim_scipy =  1 - spatial.distance.cosine(res_before, res_after)
        print('cos_sim:' + str(cos_sim_scipy))
        thresh_hold = thresh_hold
        print(res_before.shape)
        print(res_after.shape)
        try:
            np.testing.assert_allclose(res_before, res_after, atol=thresh_hold, rtol=thresh_hold)
            # return True
        except AssertionError as e:
            print(e)
    else:
        print('res_before or res_before is None!')
        print('res_before: {f}'.format(f=res_before))
        print('res_after: {f}'.format(f=res_after))
        
def get_onnx_res(path_onnx, data_input):     
    data_input = [e.cpu().numpy() for e in data_input]   
    session = ort.InferenceSession(path_onnx)
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]

    input_feed = dict(zip(input_names, data_input))
    result = session.run(output_names, input_feed)
    return result[0]
    # return dict(zip(self.output_names, result))    
        
class MLICPlusPlus(CompressionModel):
    def __init__(self, config, **kwargs):
        super().__init__(config.N, **kwargs)
        slice_num = config.slice_num
        N = config.N    # {'N': 192, 'M': 320, 'slice_num': 10, 'context_window': 5
        M = config.M
        context_window = config.context_window
        slice_num = config.slice_num
        slice_ch = M // slice_num
        assert slice_ch * slice_num == M

        self.N = N  # 192
        self.M = M  # 320
        self.context_window = context_window
        self.slice_num = slice_num
        self.slice_ch = slice_ch

        self.g_a = AnalysisTransform(N=N, M=M)  # 192,320
        self.g_s = SynthesisTransform(N=N, M=M)

        self.h_a = HyperAnalysis(M=M, N=N)
        self.h_s = HyperSynthesis(M=M, N=N)

        # Gussian Conditional
        self.gaussian_conditional = GaussianConditional(None)

        self.local_context = nn.ModuleList(
            LocalContext(dim=slice_ch)
            for _ in range(slice_num)
        )

        self.channel_context = nn.ModuleList(
            ChannelContext(in_dim=slice_ch * i, out_dim=slice_ch) if i else None
            for i in range(slice_num)
        )

        # Global Reference for non-anchors
        self.global_inter_context = nn.ModuleList(
            LinearGlobalInterContext(dim=slice_ch * i, out_dim=slice_ch * 2, num_heads=slice_ch * i // 32) if i else None
            for i in range(slice_num)
        )
        self.global_intra_context = nn.ModuleList(
            LinearGlobalIntraContext(dim=slice_ch) if i else None
            for i in range(slice_num)
        )
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch * 6, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch * 10, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2 + slice_ch * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )

        # Latent Residual Prediction
        self.lrp_anchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )
        self.lrp_nonanchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )
        
    def get_com_ga_ha(self, x):
        y = self.g_a(x) 
        z, y = self.h_a(y)
        return z, y
    

    def forward(self, x):
        """
        Using checkerboard context model with mask attention
        which divides y into anchor and non-anchor parts
        non-anchor use anchor as spatial context
        In addition, a channel-wise entropy model is used, too.
        Args:
            x: [B, 3, H, W]
        return:
            x_hat: [B, 3, H, W]
            y_likelihoods: [B, M, H // 16, W // 16]
            z_likelihoods: [B, N, H // 64, W // 64]
            likelihoods: y_likelihoods, z_likelihoods
        """
        self.update_resolutions(x.size(2) // 16, x.size(3) // 16)
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        # Hyper-parameters
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        dir_onnx = './onnx_models/'
        dir_onnx_com_ = os.path.join(dir_onnx, "decompress")
        dir_onnx_com = os.path.join(dir_onnx, "decompress")
        os.makedirs(dir_onnx_com, exist_ok=True)    
        flag_export_decompress = True
        if flag_export_decompress:
            path_onnx_sim = os.path.join(dir_onnx_com, 'decompress_hs_sim.onnx')
            print(path_onnx_sim)
            torch.onnx.export(self.h_s, 
                            z_hat,                  
                            path_onnx_sim, 
                            input_names=['z_hat'],
                            output_names=['hyper_params'],
                            opset_version=11,
                            do_constant_folding=True,
                            export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                            verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

            model_onnx = onnx.load(path_onnx_sim)
            model_simplified, check = simplify(model_onnx)
            assert check, "Simplified ONNX model could not be validated."
            onnx.save(model_simplified, path_onnx_sim)
            
            res_onnx = get_onnx_res(path_onnx_sim, [z_hat])
            get_cosine(hyper_params.cpu().detach().numpy(), res_onnx, 1e-6)
            print(get_hash(path_onnx_sim))

        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []
        y_likelihoods = []
        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                # predict residuals cause by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                # predict residuals cause by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

            else:
                global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                # predict residuals cause by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor(Use spatial context, channel context and hyper params)
                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                # predict residuals cause by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y_likelihoods": y_likelihoods, "z_likelihoods": z_likelihoods}
        }

    def update_resolutions(self, H, W):
        for i in range(len(self.global_intra_context)):
            if i == 0:
                self.local_context[i].update_resolution(H, W, next(self.parameters()).device, mask=None)
            else:
                self.local_context[i].update_resolution(H, W, next(self.parameters()).device, mask=self.local_context[0].attn_mask)

    def compress(self, x):
        torch.cuda.synchronize()
        dir_onnx = './onnx_models/'
        dir_onnx_com_ = os.path.join(dir_onnx, "compress")
        dir_onnx_com = os.path.join(dir_onnx, "compress")
        os.makedirs(dir_onnx_com, exist_ok=True)

        start_time = time.time()
        self.update_resolutions(x.size(2) // 16, x.size(3) // 16)
        y = self.g_a(x)
        z = self.h_a(y)

        import onnx
        from onnxsim import simplify
        flag_export_compress = True
        if flag_export_compress:
            path_onnx_sim = os.path.join(dir_onnx_com, 'compress_ga_sim.onnx')
            print(path_onnx_sim)
            torch.onnx.export(self.g_a, 
                            x,                  # 一个或多个张量，表示模型的输入, 多个用 tuple 表示
                            path_onnx_sim, 
                            input_names=['x'],
                            output_names=['y'],
                            opset_version=11,
                            do_constant_folding=True,
                            export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                            #use_external_data_format=False,  # use_external_data_format (可选): 当模型文件过大时，是否使用外部数据格式存储模型数据。默认为 False
                            verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

            model_onnx = onnx.load(path_onnx_sim)
            model_simplified, check = simplify(model_onnx)
            assert check, "Simplified ONNX model could not be validated."
            onnx.save(model_simplified, path_onnx_sim)
            
            res_onnx = get_onnx_res(path_onnx_sim, [x])
            get_cosine(y.cpu().detach().numpy(), res_onnx, 1e-6)
            print(get_hash(path_onnx_sim))
            
            z = self.h_a(y)
            path_onnx_sim = os.path.join(dir_onnx_com, 'compress_ha_sim.onnx')
            print(path_onnx_sim)
            torch.onnx.export(self.h_a, 
                            y,                  # 一个或多个张量，表示模型的输入, 多个用 tuple 表示
                            path_onnx_sim, 
                            input_names=['y'],
                            output_names=['z'],
                            opset_version=11,
                            do_constant_folding=True,
                            export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                            verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

            model_onnx = onnx.load(path_onnx_sim)
            model_simplified, check = simplify(model_onnx)
            assert check, "Simplified ONNX model could not be validated."
            onnx.save(model_simplified, path_onnx_sim)

            res_onnx = get_onnx_res(path_onnx_sim, [y])
            get_cosine(z.cpu().detach().numpy(), res_onnx, 1e-6)
            print(get_hash(path_onnx_sim))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)

        if flag_export_compress:
            path_onnx_sim = os.path.join(dir_onnx_com, 'compress_hs_sim.onnx')
            print(path_onnx_sim)
            torch.onnx.export(self.h_s, 
                            z_hat,                  # 一个或多个张量，表示模型的输入, 多个用 tuple 表示
                            path_onnx_sim, 
                            input_names=['z_hat'],
                            output_names=['hyper_params'],
                            opset_version=11,
                            do_constant_folding=True,
                            export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                            verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

            model_onnx = onnx.load(path_onnx_sim)
            model_simplified, check = simplify(model_onnx)
            assert check, "Simplified ONNX model could not be validated."
            onnx.save(model_simplified, path_onnx_sim)

            res_onnx = get_onnx_res(path_onnx_sim, [z_hat])
            get_cosine(hyper_params.cpu().detach().numpy(), res_onnx, 1e-6)
            print(get_hash(path_onnx_sim))

        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                print('compress for {} '.format(idx) * 10)
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                dir_onnx_com = os.path.join(dir_onnx_com, "compress_0")
                os.makedirs(dir_onnx_com, exist_ok=True)    
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'entropy_parameters_anchor_{}_sim.onnx'.format(idx))
                    print(path_onnx_sim)
                    torch.onnx.export(self.entropy_parameters_anchor[idx], 
                                    hyper_params,                  # 一个或多个张量，表示模型的输入, 多个用 tuple 表示
                                    path_onnx_sim, 
                                    input_names=['hyper_params'],
                                    output_names=['params_anchor'],
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                                    verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                        
                    res_onnx = get_onnx_res(path_onnx_sim, [hyper_params])
                    get_cosine(params_anchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                        
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # predict residuals caused by round
                # lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                input_lrp= torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                lrp_anchor = self.lrp_anchor[idx](input_lrp)
                
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'lrp_anchor_{}_sim.onnx'.format(idx))
                    print(path_onnx_sim)
                    torch.onnx.export(self.lrp_anchor[idx], 
                                    input_lrp,                  # 一个或多个张量，表示模型的输入, 多个用 tuple 表示
                                    path_onnx_sim, 
                                    input_names=['input_lrp'],
                                    output_names=['lrp_anchor'],
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                                    verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    print('export onnx success compress self.lrp_anchor{}'.format(idx)) 
                        
                    res_onnx = get_onnx_res(path_onnx_sim, [input_lrp])
                    get_cosine(lrp_anchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))

                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                # local_ctx: [B,2 * C, H, W], LocalContext 模块通过结合卷积操作和注意力机制来处理图像数据的局部上下文信息，以捕获图像中局部区域的相关特征和结构。
                local_ctx = self.local_context[idx](slice_anchor)
                
                # params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'local_context_{}_sim.onnx'.format(idx))
                    print(path_onnx_sim)
                    torch.onnx.export(self.local_context[idx], 
                                    slice_anchor,                  # 一个或多个张量，表示模型的输入, 多个用 tuple 表示
                                    path_onnx_sim, 
                                    input_names=['slice_anchor'],
                                    output_names=['local_ctx'],
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                                    verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    
                    res_onnx = get_onnx_res(path_onnx_sim, [slice_anchor])
                    get_cosine(local_ctx.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                    
                input_entropy_parameters_nonanchor = torch.cat([local_ctx, hyper_params], dim=1)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](input_entropy_parameters_nonanchor)
                
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'entropy_parameters_nonanchor_{}_sim.onnx'.format(idx))
                    print(path_onnx_sim)
                    torch.onnx.export(self.entropy_parameters_nonanchor[idx], 
                                    input_entropy_parameters_nonanchor,                  # 一个或多个张量，表示模型的输入, 多个用 tuple 表示
                                    path_onnx_sim, 
                                    input_names=['input_entropy_parameters_nonanchor'],
                                    output_names=['params_nonanchor'],
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                                    verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                       
                    res_onnx = get_onnx_res(path_onnx_sim, [input_entropy_parameters_nonanchor])
                    get_cosine(params_nonanchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))

                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                # predict residuals caused by round
                input_lrp_nonanchor = torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1)
                lrp_nonanchor = self.lrp_nonanchor[idx](input_lrp_nonanchor)
                
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'lrp_nonanchor_{}_sim.onnx'.format(idx))
                    print(path_onnx_sim)
                    torch.onnx.export(self.lrp_nonanchor[idx], 
                                    input_lrp_nonanchor,                  # 一个或多个张量，表示模型的输入, 多个用 tuple 表示
                                    path_onnx_sim, 
                                    input_names=['input_lrp_nonanchor'],
                                    output_names=['lrp_nonanchor'],
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                                    verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                        
                    res_onnx = get_onnx_res(path_onnx_sim, [input_lrp_nonanchor])
                    get_cosine(lrp_nonanchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

            else:
                print('compress for {} '.format(idx) * 10)
                # Anchor
                input_global_inter_context = torch.cat(y_hat_slices, dim=1) 
                global_inter_ctx = self.global_inter_context[idx](input_global_inter_context)
                
                dir_onnx_com = os.path.join(dir_onnx_com_, "compress_1_9")
                os.makedirs(dir_onnx_com, exist_ok=True)    
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'global_inter_context_{}_sim.onnx'.format(idx))
                    print(path_onnx_sim)
                    torch.onnx.export(self.global_inter_context[idx], 
                                    input_global_inter_context,                  # 一个或多个张量，表示模型的输入, 多个用 tuple 表示
                                    path_onnx_sim, 
                                    input_names=['input_global_inter_context'],
                                    output_names=['global_inter_ctx'],
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                                    verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                        
                    res_onnx = get_onnx_res(path_onnx_sim, [input_global_inter_context])
                    get_cosine(global_inter_ctx.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                        
                input_channel_context = torch.cat(y_hat_slices, dim=1)
                channel_ctx = self.channel_context[idx](input_channel_context)
                
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'channel_context_{}_sim.onnx'.format(idx))
                    print(path_onnx_sim)
                    torch.onnx.export(self.channel_context[idx], 
                                    input_channel_context,                  # 一个或多个张量，表示模型的输入, 多个用 tuple 表示
                                    path_onnx_sim, 
                                    input_names=['input_channel_context'],
                                    output_names=['channel_ctx'],
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                                    verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                        
                    res_onnx = get_onnx_res(path_onnx_sim, [input_channel_context])
                    get_cosine(channel_ctx.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                        
                input_entropy_parameters_anchor = torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1)                       
                params_anchor = self.entropy_parameters_anchor[idx](input_entropy_parameters_anchor)
                
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'entropy_parameters_anchor_{}_sim.onnx'.format(idx))
                    print(path_onnx_sim)
                    torch.onnx.export(self.entropy_parameters_anchor[idx], 
                                    input_entropy_parameters_anchor,                  # 一个或多个张量，表示模型的输入, 多个用 tuple 表示
                                    path_onnx_sim, 
                                    input_names=['input_entropy_parameters_anchor'],
                                    output_names=['params_anchor'],
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                                    verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    
                    res_onnx = get_onnx_res(path_onnx_sim, [input_entropy_parameters_anchor])
                    get_cosine(params_anchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                        
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # predict residuals caused by round
                
                input_lrp_anchor = torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                lrp_anchor = self.lrp_anchor[idx](input_lrp_anchor)
                
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'lrp_anchor_{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.lrp_anchor[idx], #
                                    input_lrp_anchor,   # 
                                    path_onnx_sim, 
                                    input_names=['input_lrp_anchor'],   #
                                    output_names=['lrp_anchor'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                        
                    res_onnx = get_onnx_res(path_onnx_sim, [input_lrp_anchor])
                    get_cosine(lrp_anchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                        
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'global_intra_context_{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.global_intra_context[idx],   #
                                    (y_hat_slices[-1], slice_anchor),   # 
                                    path_onnx_sim, 
                                    input_names=['y_hat_slices_last_ele, slice_anchor'],   #
                                    output_names=['global_intra_ctx'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                        
                    res_onnx = get_onnx_res(path_onnx_sim, [y_hat_slices[-1], slice_anchor])
                    get_cosine(global_intra_ctx.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                        
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'local_context_{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.local_context[idx],   #
                                    slice_anchor,   # 
                                    path_onnx_sim, 
                                    input_names=['slice_anchor'],   #
                                    output_names=['local_ctx'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                        
                    res_onnx = get_onnx_res(path_onnx_sim, [slice_anchor])
                    get_cosine(local_ctx.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                        
                input_entropy_parameters_nonanchor = torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1)                        
                params_nonanchor = self.entropy_parameters_nonanchor[idx](input_entropy_parameters_nonanchor)
                
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'entropy_parameters_nonanchor_{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.entropy_parameters_nonanchor[idx],   #
                                    input_entropy_parameters_nonanchor,   # 
                                    path_onnx_sim, 
                                    input_names=['input_entropy_parameters_nonanchor'],   #
                                    output_names=['params_nonanchor'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                        
                    res_onnx = get_onnx_res(path_onnx_sim, [input_entropy_parameters_nonanchor])
                    get_cosine(params_nonanchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                        
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                # predict residuals caused by round
                input_lrp_nonanchor = torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1)
                lrp_nonanchor = self.lrp_nonanchor[idx](input_lrp_nonanchor)
                
                if flag_export_compress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'lrp_nonanchor_{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.lrp_nonanchor[idx],   #
                                    input_lrp_nonanchor,   # 
                                    path_onnx_sim, 
                                    input_names=['input_lrp_nonanchor'],   #
                                    output_names=['lrp_nonanchor'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                        
                    res_onnx = get_onnx_res(path_onnx_sim, [input_lrp_nonanchor])
                    get_cosine(lrp_nonanchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        torch.cuda.synchronize()
        end_time = time.time()

        cost_time = end_time - start_time
        print('compress end ' * 10)
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "cost_time": cost_time
        }

    def decompress(self, strings, shape):
        torch.cuda.synchronize()
        dir_onnx = './onnx_models/'
        dir_onnx_com_ = os.path.join(dir_onnx, "decompress")
        dir_onnx_com = os.path.join(dir_onnx, "decompress")

        start_time = time.time()
        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        self.update_resolutions(z_hat.size(2) * 4, z_hat.size(3) * 4)
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for idx in range(self.slice_num):
            if idx == 0:
                print('decompress for {} '.format(idx) * 10)
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)

                dir_onnx_com = os.path.join(dir_onnx_com, "decompress_0")
                os.makedirs(dir_onnx_com, exist_ok=True) 
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'entropy_parameters_anchor_{}_sim.onnx'.format(idx)) # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.entropy_parameters_anchor[idx], 
                                    hyper_params,                  #
                                    path_onnx_sim,      
                                    input_names=['hyper_params'],      # 
                                    output_names=['params_anchor'],    # 
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)          

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    
                    res_onnx = get_onnx_res(path_onnx_sim, [hyper_params])
                    get_cosine(params_anchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))

                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                input_lrp_anchor = torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                lrp_anchor = self.lrp_anchor[idx](input_lrp_anchor)
                
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'lrp_anchor_{}_sim.onnx'.format(idx)) # 
                    torch.onnx.export(self.lrp_anchor[idx], 
                                    input_lrp_anchor,                  #
                                    path_onnx_sim,      
                                    input_names=['input_lrp_anchor'],      # 
                                    output_names=['lrp_anchor'],    # 
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)          

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    print('export onnx success decompress self.lrp_anchor_{}'.format(idx))      #
                    
                    res_onnx = get_onnx_res(path_onnx_sim, [input_lrp_anchor])
                    get_cosine(lrp_anchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                        
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'local_context_{}_sim.onnx'.format(idx)) # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.local_context[idx], 
                                    slice_anchor,                  #
                                    path_onnx_sim,      
                                    input_names=['slice_anchor'],      # 
                                    output_names=['local_ctx'],    # 
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)          

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                        
                    res_onnx = get_onnx_res(path_onnx_sim, [slice_anchor])
                    get_cosine(local_ctx.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                        
                input_par = torch.cat([local_ctx, hyper_params], dim=1)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](input_par)
                
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'entropy_parameters_nonanchor_{}_sim.onnx'.format(idx)) # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.entropy_parameters_nonanchor[idx], 
                                    input_par,                  #
                                    path_onnx_sim,      
                                    input_names=['input_par'],      # 
                                    output_names=['params_nonanchor'],    # 
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)          

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)

                    res_onnx = get_onnx_res(path_onnx_sim, [input_par])
                    get_cosine(params_nonanchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                input_lrp_ = torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1)
                lrp_nonanchor = self.lrp_nonanchor[idx](input_lrp_)
                
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'lrp_nonanchor_{}_sim.onnx'.format(idx))     # 
                    torch.onnx.export(self.lrp_nonanchor[idx], 
                                    input_lrp_,                  #
                                    path_onnx_sim,      
                                    input_names=['input_lrp_'],      # 
                                    output_names=['lrp_nonanchor'],    # 
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)          

                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)

                    res_onnx = get_onnx_res(path_onnx_sim, [input_lrp_])
                    get_cosine(lrp_nonanchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                        
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

            else:
                print('decompress for {} '.format(idx) * 10)
                # Anchor
                input_global_inter_context = torch.cat(y_hat_slices, dim=1)
                global_inter_ctx = self.global_inter_context[idx](input_global_inter_context)
                
                dir_onnx_com = os.path.join(dir_onnx_com_, "decompress_1_9")
                os.makedirs(dir_onnx_com, exist_ok=True) 
                
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'global_inter_context_{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.global_inter_context[idx],   #
                                    input_global_inter_context,   # 
                                    path_onnx_sim, 
                                    input_names=['input_global_inter_context'],   #
                                    output_names=['global_inter_ctx'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           
                    
                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    
                    res_onnx = get_onnx_res(path_onnx_sim, [input_global_inter_context])
                    get_cosine(global_inter_ctx.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                
                input_channel_context = torch.cat(y_hat_slices, dim=1)
                channel_ctx = self.channel_context[idx](input_channel_context)
                
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'channel_context_{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.channel_context[idx],   #
                                    input_channel_context,   # 
                                    path_onnx_sim, 
                                    input_names=['input_channel_context'],   #
                                    output_names=['channel_ctx'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           
                    
                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    
                    res_onnx = get_onnx_res(path_onnx_sim, [input_channel_context])
                    get_cosine(channel_ctx.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                    
                input_entropy_parameters_anchor = torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1)    
                params_anchor = self.entropy_parameters_anchor[idx](input_entropy_parameters_anchor)
                
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'entropy_parameters_anchor{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.entropy_parameters_anchor[idx],   #
                                    input_entropy_parameters_anchor,   # 
                                    path_onnx_sim, 
                                    input_names=['input_entropy_parameters_anchor'],   #
                                    output_names=['params_anchor'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           
                    
                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    
                    res_onnx = get_onnx_res(path_onnx_sim, [input_entropy_parameters_anchor])
                    get_cosine(params_anchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                    
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                input_lrp_anchor = torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)                
                lrp_anchor = self.lrp_anchor[idx](input_lrp_anchor)
                
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'lrp_anchor_{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.lrp_anchor[idx],   #
                                    input_lrp_anchor,   # 
                                    path_onnx_sim, 
                                    input_names=['input_lrp_anchor'],   #
                                    output_names=['lrp_anchor'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           
                    
                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    
                    res_onnx = get_onnx_res(path_onnx_sim, [input_lrp_anchor])
                    get_cosine(lrp_anchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # Non-anchor
                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'global_intra_context_{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.global_intra_context[idx],   #
                                    (y_hat_slices[-1], slice_anchor),   # 
                                    path_onnx_sim, 
                                    input_names=['y_hat_slices_last', 'slice_anchor'],   #
                                    output_names=['global_intra_ctx'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           
                    
                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    
                    res_onnx = get_onnx_res(path_onnx_sim, [y_hat_slices[-1], slice_anchor])
                    get_cosine(global_intra_ctx.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                    
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'local_context_{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.local_context[idx],   #
                                    slice_anchor,   # 
                                    path_onnx_sim, 
                                    input_names=['input_local_context'],   #
                                    output_names=['global_intra_ctx'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           
                    
                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    
                    res_onnx = get_onnx_res(path_onnx_sim, [slice_anchor])
                    get_cosine(local_ctx.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                
                input_entropy_parameters_nonanchor = torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](input_entropy_parameters_nonanchor)
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'entropy_parameters_nonanchor_{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.entropy_parameters_nonanchor[idx],   #
                                    input_entropy_parameters_nonanchor,   # 
                                    path_onnx_sim, 
                                    input_names=['input_entropy_parameters_nonanchor'],   #
                                    output_names=['params_nonanchor'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           
                    
                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    
                    res_onnx = get_onnx_res(path_onnx_sim, [input_entropy_parameters_nonanchor])
                    get_cosine(params_nonanchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                    
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                input_lrp_nonanchor = torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1)
                lrp_nonanchor = self.lrp_nonanchor[idx](input_lrp_nonanchor)
                
                if flag_export_decompress:
                    path_onnx_sim = os.path.join(dir_onnx_com, 'lrp_nonanchor_{}_sim.onnx'.format(idx))    # 
                    print(path_onnx_sim)
                    torch.onnx.export(self.lrp_nonanchor[idx],   #
                                    input_lrp_nonanchor,   # 
                                    path_onnx_sim, 
                                    input_names=['input_lrp_nonanchor'],   #
                                    output_names=['output_lrp_nonanchor'],    #
                                    opset_version=11,
                                    do_constant_folding=True,
                                    export_params=True,     
                                    verbose=False)           
                    
                    model_onnx = onnx.load(path_onnx_sim)
                    model_simplified, check = simplify(model_onnx)
                    assert check, "Simplified ONNX model could not be validated."
                    onnx.save(model_simplified, path_onnx_sim)
                    
                    res_onnx = get_onnx_res(path_onnx_sim, [input_lrp_nonanchor])
                    get_cosine(lrp_nonanchor.cpu().detach().numpy(), res_onnx, 1e-6)
                    print(get_hash(path_onnx_sim))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        y_hat = torch.cat(y_hat_slices, dim=1) # y_hat: (1, 320, 32, 48)
        x_hat = self.g_s(y_hat) # x_hat: (1, 3, 512, 768)
        if flag_export_decompress:
            path_onnx = os.path.join(dir_onnx_com_, 'decompress_gs_sim.onnx')
            print(path_onnx)
            torch.onnx.export(self.g_s, 
                            y_hat,                  
                            path_onnx, 
                            input_names=['y_hat'],
                            output_names=['x_hat'],
                            opset_version=11,
                            do_constant_folding=True,
                            export_params=True,     # export_params (可选): 是否导出模型中的参数。默认为 True
                            verbose=False)           # verbose (可选): 是否打印详细的信息, 默认为 False

            model_onnx = onnx.load(path_onnx)
            model_simplified, check = simplify(model_onnx)
            assert check, "Simplified ONNX model could not be validated."
            onnx.save(model_simplified, path_onnx)
            
            res_onnx = get_onnx_res(path_onnx, [y_hat])
            get_cosine(x_hat.cpu().detach().numpy(), res_onnx, 1e-6)
            print(get_hash(path_onnx))

        # torch.cuda.synchronize()
        end_time = time.time()

        cost_time = end_time - start_time
        print('decompress end ' * 10)
        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
