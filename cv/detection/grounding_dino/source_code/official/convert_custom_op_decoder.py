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

graph = gs.import_onnx(onnx.load("weights/decoder_groundingdino_swint_ogc.onnx"))
out_filename = "weights/decoder_groundingdino_swint_ogc_custom_op.onnx"

deform_attn_core_encoder_in_map = [
    ['/transformer/encoder/layers.0/self_attn/Reshape_output_0','/transformer/encoder/layers.0/self_attn/Add_output_0', '/transformer/encoder/layers.0/self_attn/Softmax_output_0'],
    ['/transformer/encoder/layers.1/self_attn/Reshape_output_0','/transformer/encoder/layers.1/self_attn/Add_output_0', '/transformer/encoder/layers.1/self_attn/Softmax_output_0'],
    ['/transformer/encoder/layers.2/self_attn/Reshape_output_0','/transformer/encoder/layers.2/self_attn/Add_output_0', '/transformer/encoder/layers.2/self_attn/Softmax_output_0'],
    ['/transformer/encoder/layers.3/self_attn/Reshape_output_0','/transformer/encoder/layers.3/self_attn/Add_output_0', '/transformer/encoder/layers.3/self_attn/Softmax_output_0'],
    ['/transformer/encoder/layers.4/self_attn/Reshape_output_0','/transformer/encoder/layers.4/self_attn/Add_output_0', '/transformer/encoder/layers.4/self_attn/Softmax_output_0'],
    ['/transformer/encoder/layers.5/self_attn/Reshape_output_0','/transformer/encoder/layers.5/self_attn/Add_output_0', '/transformer/encoder/layers.5/self_attn/Softmax_output_0']
]



deform_attn_core_encoder_out_map = [
    ['/transformer/encoder/layers.0/self_attn/ReduceSum_output_0'],
    ['/transformer/encoder/layers.1/self_attn/ReduceSum_output_0'],
    ['/transformer/encoder/layers.2/self_attn/ReduceSum_output_0'],
    ['/transformer/encoder/layers.3/self_attn/ReduceSum_output_0'],
    ['/transformer/encoder/layers.4/self_attn/ReduceSum_output_0'],
    ['/transformer/encoder/layers.5/self_attn/ReduceSum_output_0']
]

deform_attn_core_decoder_in_map = [
    ['/transformer/decoder/layers.0/cross_attn/Reshape_output_0','/transformer/decoder/layers.0/cross_attn/Add_output_0', '/transformer/decoder/layers.0/cross_attn/Softmax_output_0'],
    ['/transformer/decoder/layers.1/cross_attn/Reshape_output_0','/transformer/decoder/layers.1/cross_attn/Add_output_0', '/transformer/decoder/layers.1/cross_attn/Softmax_output_0'],
    ['/transformer/decoder/layers.2/cross_attn/Reshape_output_0','/transformer/decoder/layers.2/cross_attn/Add_output_0', '/transformer/decoder/layers.2/cross_attn/Softmax_output_0'],
    ['/transformer/decoder/layers.3/cross_attn/Reshape_output_0','/transformer/decoder/layers.3/cross_attn/Add_output_0', '/transformer/decoder/layers.3/cross_attn/Softmax_output_0'],
    ['/transformer/decoder/layers.4/cross_attn/Reshape_output_0','/transformer/decoder/layers.4/cross_attn/Add_output_0', '/transformer/decoder/layers.4/cross_attn/Softmax_output_0'],
    ['/transformer/decoder/layers.5/cross_attn/Reshape_output_0','/transformer/decoder/layers.5/cross_attn/Add_output_0', '/transformer/decoder/layers.5/cross_attn/Softmax_output_0']
]



deform_attn_core_decoder_out_map = [
    ['/transformer/decoder/layers.0/cross_attn/ReduceSum_output_0'],
    ['/transformer/decoder/layers.1/cross_attn/ReduceSum_output_0'],
    ['/transformer/decoder/layers.2/cross_attn/ReduceSum_output_0'],
    ['/transformer/decoder/layers.3/cross_attn/ReduceSum_output_0'],
    ['/transformer/decoder/layers.4/cross_attn/ReduceSum_output_0'],
    ['/transformer/decoder/layers.5/cross_attn/ReduceSum_output_0']
]


# sine_position_embed_in_name = '/transformer/encoder/Unsqueeze_output_0'
sine_position_embed_in_name = '/transformer/encoder/Unsqueeze_10_output_0'


sine_position_embed_out_name = '/transformer/encoder/Reshape_output_0'


query_sine_embed_in_map = [
    ['/transformer/decoder/Gather_output_0'],
    ['/transformer/decoder/Gather_5_output_0'],
    ['/transformer/decoder/Gather_10_output_0'],
    ['/transformer/decoder/Gather_15_output_0'],
    ['/transformer/decoder/Gather_20_output_0'],
    ['/transformer/decoder/Gather_25_output_0']
]



# query_sine_embed_out_map = [
#     ['/transformer/decoder/Concat_8_output_0'],
#     ['/transformer/decoder/Concat_17_output_0'],
#     ['/transformer/decoder/Concat_26_output_0'],
#     ['/transformer/decoder/Concat_35_output_0'],
#     ['/transformer/decoder/Concat_44_output_0'],
#     ['/transformer/decoder/Concat_53_output_0']
# ]

query_sine_embed_out_map = [
    ['/transformer/decoder/Concat_9_output_0'],
    ['/transformer/decoder/Concat_18_output_0'],
    ['/transformer/decoder/Concat_27_output_0'],
    ['/transformer/decoder/Concat_36_output_0'],
    ['/transformer/decoder/Concat_45_output_0'],
    ['/transformer/decoder/Concat_54_output_0']
]

@gs.Graph.register()
def vastai_deform_attn_core(self, in_tensors, out_tensors, name):
    # Note: seems directly pass np array to attrs is not Allowed
    # Need to covert to list, and only support 1D (ONNX AttributeProto limit)
    attrs = {"input_format": ["UNRESTRICTED", "UNRESTRICTED", "UNRESTRICTED"],
    "output_format": ["UNRESTRICTED"],
    "batch": 8,
    "split_num" : 4,
    "enable_parallel": True,
    "input_core_split": [1, 0, 1, 0, 1, 0], #core split factor = 16 at in channel axis = 1
    "output_core_split": [1, 0], #core split factor = 16 at out channel axis = 2
    "input_from_ddr": [1, 1, 1],
    "split_h" : [100, 50, 25, 13],
    "split_w" : [167, 84, 42, 21],
    "output_to_ddr" : [1],
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
def vastai_sine_position_embed(self, in_tensors, out_tensors, name):
    # Note: seems directly pass np array to attrs is not Allowed
    # Need to covert to list, and only support 1D (ONNX AttributeProto limit)
    attrs = {"input_format": ["YX"],
    "output_format": ["YX"],
    "num_pos_feats": 256,
    "exchange_xy" : False,
    "enable_parallel": False,
    "input_core_split": [16, 1], #core split factor = 16 at in channel axis = 1
    "output_core_split": [16, 1], #core split factor = 16 at out channel axis = 2
    "input_from_ddr": [1]
    # "output_to_ddr": [1]
    }
    # Note:
    # default return is a list, can also return a single output by [index]

    return self.layer(op="CustomOp",
        name=name,
        inputs=[in_tensors[0]],
        outputs=out_tensors,
        attrs=attrs)

@gs.Graph.register()
def vastai_query_sine_embed(self, in_tensors, out_tensors, name):
    # Note: seems directly pass np array to attrs is not Allowed
    # Need to covert to list, and only support 1D (ONNX AttributeProto limit)
    attrs = {"input_format": ["YX"],
    "output_format": ["YX"],
    "enable_parallel": False,
    "input_core_split": [16, 0], #core split factor = 16 at in channel axis = 1
    "output_core_split": [16, 0], #core split factor = 16 at out channel axis = 2
    "input_from_ddr": [0]
    }
    # Note:
    # default return is a list, can also return a single output by [index]

    return self.layer(op="CustomOp",
        name=name,
        inputs=[in_tensors[0]],
        outputs=out_tensors,
        attrs=attrs)

tmap = graph.tensors()

idx = 0
for in_names, out_names in zip(deform_attn_core_encoder_in_map, deform_attn_core_encoder_out_map):
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

##########################################################################################
# idx = 0
for in_names, out_names in zip(deform_attn_core_decoder_in_map, deform_attn_core_decoder_out_map):
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
    # mid10 = gs.Variable(name=f"deform_attn_{idx}_mid10", dtype=np.float32, shape=(in_tensors[1].shape[1], 8, in_tensors[1].shape[2]//8))
    # mid11 = gs.Variable(name=f"deform_attn_{idx}_mid11", dtype=np.float32, shape=(8, in_tensors[1].shape[1], in_tensors[1].shape[2]//8))
    mid10 = gs.Variable(name=f"deform_attn_{idx}_mid10", dtype=np.float32, shape=(in_tensors[1].shape[1], 8, in_tensors[1].shape[3]*in_tensors[1].shape[4]*in_tensors[1].shape[5]))
    mid11 = gs.Variable(name=f"deform_attn_{idx}_mid11", dtype=np.float32, shape=(8, in_tensors[1].shape[1], in_tensors[1].shape[3]*in_tensors[1].shape[4]*in_tensors[1].shape[5]))
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
##########################################################################################

in_name = sine_position_embed_in_name
out_name = sine_position_embed_out_name
in_tensors = []
tmap[in_name].outputs.clear()
in_tensors.append(tmap[in_name])
out_tensors = []
tmap[out_name].inputs.clear()
out_tensors.append(tmap[out_name])

name = f'VASTAI_GET_SINE_POS_EMBED'
position_embed = graph.vastai_sine_position_embed(in_tensors, out_tensors, name)

for in_names, out_names in zip(query_sine_embed_in_map, query_sine_embed_out_map):
    in_tensors = []
    for in_name in in_names:
        tmap[in_name].outputs.clear()
        in_tensors.append(tmap[in_name])
    out_tensors = []
    for out_name in out_names:
        tmap[out_name].inputs.clear()
        out_tensors.append(tmap[out_name])

    name = f'VASTAI_GEN_SINEEMBED_FOR_POSITION'
    query_embed = graph.vastai_query_sine_embed(in_tensors, out_tensors, name)
    
graph.cleanup().toposort()

for out in graph.outputs:
    out.dtype = np.float32
onnx.save(gs.export_onnx(graph), out_filename, save_as_external_data=False, all_tensors_to_one_file=False)

