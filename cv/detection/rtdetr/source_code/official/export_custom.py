import os
import sys
import onnx_graphsurgeon as gs
import onnx
import numpy as np

# in_filename = "./code/model_check/vit_custom_models/weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.onnx"
# out_filename = "./code/model_check/vit_custom_models/weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_custom.onnx"

in_filename = sys.argv[1]
out_filename = sys.argv[2]

graph = gs.import_onnx(onnx.load(in_filename))



deform_attn_core_in_map = [
    ['/model/decoder/decoder/layers.0/cross_attn/Reshape_output_0', '/model/decoder/decoder/layers.0/cross_attn/Add_output_0', '/model/decoder/decoder/layers.0/cross_attn/Softmax_output_0'],
    ['/model/decoder/decoder/layers.1/cross_attn/Reshape_output_0', '/model/decoder/decoder/layers.1/cross_attn/Add_output_0', '/model/decoder/decoder/layers.1/cross_attn/Softmax_output_0'],
    ['/model/decoder/decoder/layers.2/cross_attn/Reshape_output_0', '/model/decoder/decoder/layers.2/cross_attn/Add_output_0', '/model/decoder/decoder/layers.2/cross_attn/Softmax_output_0'],
   ]

# deform_attn_core_in_map = [
#     ['/sem_seg_head/transformer/encoder/layers.0/self_attn/Reshape_output_0','/sem_seg_head/transformer/encoder/layers.0/self_attn/Add_output_0', '/sem_seg_head/transformer/encoder/layers.0/self_attn/Reshape_15_output_0'],
#     ['/sem_seg_head/transformer/encoder/layers.1/self_attn/Reshape_output_0','/sem_seg_head/transformer/encoder/layers.1/self_attn/Add_output_0', '/sem_seg_head/transformer/encoder/layers.1/self_attn/Reshape_15_output_0'],
#     ['/sem_seg_head/transformer/encoder/layers.2/self_attn/Reshape_output_0','/sem_seg_head/transformer/encoder/layers.2/self_attn/Add_output_0', '/sem_seg_head/transformer/encoder/layers.2/self_attn/Reshape_15_output_0'],
#     ['/sem_seg_head/transformer/encoder/layers.3/self_attn/Reshape_output_0','/sem_seg_head/transformer/encoder/layers.3/self_attn/Add_output_0', '/sem_seg_head/transformer/encoder/layers.3/self_attn/Reshape_15_output_0'],
#     ['/sem_seg_head/transformer/encoder/layers.4/self_attn/Reshape_output_0','/sem_seg_head/transformer/encoder/layers.4/self_attn/Add_output_0', '/sem_seg_head/transformer/encoder/layers.4/self_attn/Reshape_15_output_0'],
#     ['/sem_seg_head/transformer/encoder/layers.5/self_attn/Reshape_output_0','/sem_seg_head/transformer/encoder/layers.5/self_attn/Add_output_0', '/sem_seg_head/transformer/encoder/layers.5/self_attn/Reshape_15_output_0']
# ]




deform_attn_core_out_map = [
    ['/model/decoder/decoder/layers.0/cross_attn/ReduceSum_output_0'],
    ['/model/decoder/decoder/layers.1/cross_attn/ReduceSum_output_0'],
    ['/model/decoder/decoder/layers.2/cross_attn/ReduceSum_output_0'],
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
    "split_h" : [80, 40, 20],
    "split_w" : [80, 40, 20],
    "input_from_ddr" : [1,1,1],
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

tmap = graph.tensors()
idx = 0
for in_names, out_names in zip(deform_attn_core_in_map, deform_attn_core_out_map):
    in_tensors = []
    for in_name in in_names:
        tmap[in_name].outputs.clear()
        in_tensors.append(tmap[in_name])
    out_tensors = []
    for out_name in out_names:
        tmap[out_name].inputs.clear()
        out_tensors.append(tmap[out_name])

    print('in_tensors[0]', in_tensors[0].shape)
    print('in_tensors[1]', in_tensors[1].shape)
    print('in_tensors[2]', in_tensors[2].shape)
    # trans for input0
    mid00 = gs.Variable(name=f"deform_attn_{idx}_mid00", dtype=np.float32, shape=(in_tensors[0].shape[1],in_tensors[0].shape[2],in_tensors[0].shape[3]))
    mid01 = gs.Variable(name=f"deform_attn_{idx}_mid01", dtype=np.float32, shape=(in_tensors[0].shape[2],in_tensors[0].shape[1],in_tensors[0].shape[3]))
    reshape0 = graph.vastai_deform_reshape0(in_tensors[0], mid00, f"deform_atten_{idx}_reshape0")
    transpose0 = graph.vastai_deform_transpose102(mid00, mid01, f"deform_atten_{idx}_transpose0")

    # trans for input1
    size_last = in_tensors[1].shape[2]//8
    if len(in_tensors[1].shape)>3:
        for si in range(3,len(in_tensors[1].shape)):
            size_last *= in_tensors[1].shape[si]
        
    mid10 = gs.Variable(name=f"deform_attn_{idx}_mid10", dtype=np.float32, shape=(in_tensors[1].shape[1], 8, size_last))
    mid11 = gs.Variable(name=f"deform_attn_{idx}_mid11", dtype=np.float32, shape=(8, in_tensors[1].shape[1], size_last))
    reshape1 = graph.vastai_deform_reshape0(in_tensors[1], mid10, f"deform_atten_{idx}_reshape1")
    transpose1 = graph.vastai_deform_transpose102(mid10, mid11, f"deform_atten_{idx}_transpose1")

    # trans for input2
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

graph.cleanup().toposort()

for out in graph.outputs:
    out.dtype = np.float32
    
# onnx.save(gs.export_onnx(graph), out_filename, save_as_external_data=True, all_tensors_to_one_file=True)
onnx.save(gs.export_onnx(graph), out_filename)

