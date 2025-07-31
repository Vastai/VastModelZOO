# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import onnx_graphsurgeon as gs
import onnx
import numpy as np
import math

# in_filename = "mask2former.onnx"
import sys
# in_filename = './code/model_check/vit_custom_models/source/Mask2Former/mask2former_sim.onnx'# sys.argv[1]
# out_filename = "./code/model_check/vit_custom_models/source/Mask2Former/mask2former_sim_with_custom.onnx"

in_filename = sys.argv[1]
out_filename = sys.argv[2]


deform_attn_core_in_map = [
    ['/sem_seg_head/transformer/encoder/layers.0/self_attn/Reshape_output_0','/sem_seg_head/transformer/encoder/layers.0/self_attn/Add_output_0', '/sem_seg_head/transformer/encoder/layers.0/self_attn/Softmax_output_0'],
    ['/sem_seg_head/transformer/encoder/layers.1/self_attn/Reshape_output_0','/sem_seg_head/transformer/encoder/layers.1/self_attn/Add_output_0', '/sem_seg_head/transformer/encoder/layers.1/self_attn/Softmax_output_0'],
    ['/sem_seg_head/transformer/encoder/layers.2/self_attn/Reshape_output_0','/sem_seg_head/transformer/encoder/layers.2/self_attn/Add_output_0', '/sem_seg_head/transformer/encoder/layers.2/self_attn/Softmax_output_0'],
    ['/sem_seg_head/transformer/encoder/layers.3/self_attn/Reshape_output_0','/sem_seg_head/transformer/encoder/layers.3/self_attn/Add_output_0', '/sem_seg_head/transformer/encoder/layers.3/self_attn/Softmax_output_0'],
    ['/sem_seg_head/transformer/encoder/layers.4/self_attn/Reshape_output_0','/sem_seg_head/transformer/encoder/layers.4/self_attn/Add_output_0', '/sem_seg_head/transformer/encoder/layers.4/self_attn/Softmax_output_0'],
    ['/sem_seg_head/transformer/encoder/layers.5/self_attn/Reshape_output_0','/sem_seg_head/transformer/encoder/layers.5/self_attn/Add_output_0', '/sem_seg_head/transformer/encoder/layers.5/self_attn/Softmax_output_0']
]



deform_attn_core_out_map = [
    ['/sem_seg_head/transformer/encoder/layers.0/self_attn/ReduceSum_output_0'],
    ['/sem_seg_head/transformer/encoder/layers.1/self_attn/ReduceSum_output_0'],
    ['/sem_seg_head/transformer/encoder/layers.2/self_attn/ReduceSum_output_0'],
    ['/sem_seg_head/transformer/encoder/layers.3/self_attn/ReduceSum_output_0'],
    ['/sem_seg_head/transformer/encoder/layers.4/self_attn/ReduceSum_output_0'],
    ['/sem_seg_head/transformer/encoder/layers.5/self_attn/ReduceSum_output_0']
]

fph_in_map = [
    ['/sem_seg_head/predictor/MatMul_output_0'],
    ['/sem_seg_head/predictor/MatMul_1_output_0'],
    ['/sem_seg_head/predictor/MatMul_2_output_0'],
    ['/sem_seg_head/predictor/MatMul_3_output_0'],
    ['/sem_seg_head/predictor/MatMul_4_output_0'],
    ['/sem_seg_head/predictor/MatMul_5_output_0'],
    ['/sem_seg_head/predictor/MatMul_6_output_0'],
    ['/sem_seg_head/predictor/MatMul_7_output_0'],
    ['/sem_seg_head/predictor/MatMul_8_output_0']
]
fph_out_map = [
    ['/sem_seg_head/predictor/transformer_cross_attention_layers.0/multihead_attn/Where_output_0'],
    ['/sem_seg_head/predictor/transformer_cross_attention_layers.1/multihead_attn/Where_output_0'],
    ['/sem_seg_head/predictor/transformer_cross_attention_layers.2/multihead_attn/Where_output_0'],
    ['/sem_seg_head/predictor/transformer_cross_attention_layers.3/multihead_attn/Where_output_0'],
    ['/sem_seg_head/predictor/transformer_cross_attention_layers.4/multihead_attn/Where_output_0'],
    ['/sem_seg_head/predictor/transformer_cross_attention_layers.5/multihead_attn/Where_output_0'],
    ['/sem_seg_head/predictor/transformer_cross_attention_layers.6/multihead_attn/Where_output_0'],
    ['/sem_seg_head/predictor/transformer_cross_attention_layers.7/multihead_attn/Where_output_0'],
    ['/sem_seg_head/predictor/transformer_cross_attention_layers.8/multihead_attn/Where_output_0']
]

@gs.Graph.register()
def vastai_deform_attn_core(self, in_tensors, out_tensors, name):
    # Note: seems directly pass np array to attrs is not Allowed
    # Need to covert to list, and only support 1D (ONNX AttributeProto limit)
    attrs = {"input_format": ["UNRESTRICTED", "UNRESTRICTED", "UNRESTRICTED"],
    "output_format": ["UNRESTRICTED"],
    "batch": 8,
    "split_num" : 3,
    "enable_parallel": True,
    "input_core_split": [1, 0, 1, 0, 1, 0], #core split factor = 16 at in channel axis = 1
    "output_core_split": [1, 0], #core split factor = 16 at out channel axis = 2
    "input_from_ddr": [1, 1, 1],
    "output_to_ddr": [1],
    "split_h" : [32, 64, 128],
    "split_w" : [32, 64, 128]
    # "real_in_shape": in_tensor.shape
    }
    # Note:
    # default return is a list, can also return a single output by [index]

    return self.layer(op="CustomOp",
        name=name,
        inputs=[in_tensors[0], in_tensors[1], in_tensors[2]],
        outputs=out_tensors,
        attrs=attrs)

@gs.Graph.register()
def vastai_forward_prediction_heads(self, in_tensors, out_tensors, name):
    # Note: seems directly pass np array to attrs is not Allowed
    # Need to covert to list, and only support 1D (ONNX AttributeProto limit)
    print(in_tensors[0].shape)
    print(out_tensors[0].shape)
    scale = math.sqrt(in_tensors[0].shape[1] * in_tensors[0].shape[2] / out_tensors[0].shape[2])
    attrs = {"input_format": ["UNRESTRICTED"],
    "output_format": ["UNRESTRICTED"],
    "enable_parallel": False,
    "input_core_split": [1, 0], #core split factor = 16 at in channel axis = 1
    "output_core_split": [1, 0], #core split factor = 16 at out channel axis = 2
    "output_to_ddr": [1],
    "scale_h" : scale,
    "scale_w" : scale,
    "tile_num" : out_tensors[0].shape[0],
    "replace_val" : -60000.0,
    "align_corners" : 0,
    "method" : 0,
    "nearest_mode" : "floor",
    "mode" : "linear",
    "cubic_coeff_a" : -0.75,
    "coordinate_transformation_mode": "half_pixel"
    # "real_in_shape": in_tensor.shape
    }
    # Note:
    # default return is a list, can also return a single output by [index]

    return self.layer(op="CustomOp",
        name=name,
        inputs=in_tensors,
        outputs=out_tensors,
        attrs=attrs)

@gs.Graph.register()
def vastai_deform_reshape0(self, in_tensor, out_tensor, name):
    new_shape = gs.Constant(f"{name}_new_shape", values=np.int64(out_tensor.shape))
    return self.layer(op="Reshape",
        #attrs={"shape" : out_tensor.shape},
        inputs=[in_tensor, new_shape],
        outputs=[out_tensor],
        name = name)

@gs.Graph.register()
def vastai_deform_transpose102(self, in_tensor, out_tensor, name):
    return self.layer(op="Transpose",
        attrs={"perm":[1,0,2]},
        inputs=[in_tensor],
        outputs=[out_tensor],
        name=name)

@gs.Graph.register()
def vastai_concat_axis0(self, in_tensors, out_tensors, name):
    return self.layer(op="Concat",
        attrs={"axis":0},
        inputs=in_tensors,
        outputs=out_tensors,
        name=name)

graph = gs.import_onnx(onnx.load(in_filename))
tmap = graph.tensors()
idx = 0
for in_names, out_names in zip(deform_attn_core_in_map, deform_attn_core_out_map):
    if out_names[0] not in tmap:
        continue
    in_tensors = []
    for in_name in in_names:
        tmap[in_name].outputs.clear()
        in_tensors.append(tmap[in_name])
    out_tensors = []
    for out_name in out_names:
        tmap[out_name].inputs.clear()
        out_tensors.append(tmap[out_name])

    # trans for input0
    mid00 = gs.Variable(name=f"deform_attn_{idx}_mid00", dtype=np.float32, shape=(in_tensors[0].shape[1],in_tensors[0].shape[2],in_tensors[0].shape[3]))
    mid01 = gs.Variable(name=f"deform_attn_{idx}_mid01", dtype=np.float32, shape=(in_tensors[0].shape[2],in_tensors[0].shape[1],in_tensors[0].shape[3]))
    reshape0 = graph.vastai_deform_reshape0(in_tensors[0], mid00, f"deform_atten_{idx}_reshape0")
    transpose0 = graph.vastai_deform_transpose102(mid00, mid01, f"deform_atten_{idx}_transpose0")

    # trans for input1
    mid10 = gs.Variable(name=f"deform_attn_{idx}_mid10", dtype=np.float32, shape=(in_tensors[1].shape[1], 8, in_tensors[1].shape[2]//8))
    mid11 = gs.Variable(name=f"deform_attn_{idx}_mid11", dtype=np.float32, shape=(8, in_tensors[1].shape[1], in_tensors[1].shape[2]//8))
    reshape1 = graph.vastai_deform_reshape0(in_tensors[1], mid10, f"deform_atten_{idx}_reshape1")
    transpose1 = graph.vastai_deform_transpose102(mid10, mid11, f"deform_atten_{idx}_transpose1")

    # trans for input0
    mid20 = gs.Variable(name=f"deform_attn_{idx}_mid20", dtype=np.float32, shape=(in_tensors[2].shape[1],in_tensors[2].shape[2],in_tensors[2].shape[3]))
    mid21 = gs.Variable(name=f"deform_attn_{idx}_mid21", dtype=np.float32, shape=(in_tensors[2].shape[2],in_tensors[2].shape[1],in_tensors[2].shape[3]))
    reshape2 = graph.vastai_deform_reshape0(in_tensors[2], mid20, f"deform_atten_{idx}_reshape2")
    transpose2 = graph.vastai_deform_transpose102(mid20, mid21, f"deform_atten_{idx}_transpose2")

    name = f'VASTAI_OP_DEFORM_ATTN_CORE'
    unfold = graph.vastai_deform_attn_core([mid01, mid11, mid21], out_tensors, name)
    idx += 1
    # unfold[0].shape = tmap[out_name].shape
    # unfold[0].dtype = tmap[out_name].dtype
    # for i in range(len(tmap[out_name].outputs)):
    #   unfold[0].outputs.append(tmap[out_name].outputs[i])
    # tmap[out_name].outputs.clear()

idx = 0
for in_names, out_names in zip(fph_in_map, fph_out_map):
    if out_names[0] not in tmap:
        continue
    in_tensors = []
    for in_name in in_names:
        tmap[in_name].outputs.clear()
        in_tensors.append(tmap[in_name])
    out_tensors = []
    for out_name in out_names:
        tmap[out_name].inputs.clear()
        out_tensors.append(tmap[out_name])

    name = f'VASTAI_OP_FORWARD_PREDICTION_HEADS'

    mid = gs.Variable(name=f"fph_{idx}_mid", dtype=np.float32, shape=(in_tensors[0].shape[1],int(math.sqrt(in_tensors[0].shape[2])),int(math.sqrt(in_tensors[0].shape[2]))))
    reshape0 = graph.vastai_deform_reshape0(in_tensors[0], mid, f"fph_{idx}_reshape0")

    unfold = graph.vastai_forward_prediction_heads([mid], out_tensors, name)

    # mid00 = gs.Variable(name=f"fph_{idx}_mid00", dtype=np.float32, shape=(in_tensors[0].shape[1],int(math.sqrt(in_tensors[0].shape[2])),int(math.sqrt(in_tensors[0].shape[2]))))
    # reshape0 = graph.vastai_deform_reshape0(in_tensors[0], mid00, f"fph_{idx}_reshape0")
    # mid01 = gs.Variable(name=f"fph_{idx}_mid01", dtype=np.float32, shape=(out_tensors[0].shape[0]//2,out_tensors[0].shape[1],out_tensors[0].shape[2]))
    # unfold0 = graph.vastai_forward_prediction_heads([mid00], [mid01], name)

    # mid10 = gs.Variable(name=f"fph_{idx}_mid10", dtype=np.float32, shape=(in_tensors[0].shape[1],int(math.sqrt(in_tensors[0].shape[2])),int(math.sqrt(in_tensors[0].shape[2]))))
    # reshape1 = graph.vastai_deform_reshape0(in_tensors[0], mid10, f"fph_{idx}_reshape1")
    # mid11 = gs.Variable(name=f"fph_{idx}_mid11", dtype=np.float32, shape=(out_tensors[0].shape[0]//2,out_tensors[0].shape[1],out_tensors[0].shape[2]))
    # unfold1 = graph.vastai_forward_prediction_heads([mid00], [mid11], name)

    # concat = graph.vastai_concat_axis0([mid01, mid11], out_tensors, f"fph_concat_{idx}")
    idx += 1

graph.cleanup().toposort()
graph.cleanup()

for out in graph.outputs:
    out.dtype = np.float32
onnx.save(gs.export_onnx(graph), out_filename)

