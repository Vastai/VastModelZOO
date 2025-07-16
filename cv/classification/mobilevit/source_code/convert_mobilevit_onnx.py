import onnx_graphsurgeon as gs
import onnx
from onnxsim import simplify
import numpy as np
import os

@gs.Graph.register()
def vastai_unfold(self, in_tensor, out_tensor, name):
    # Note: seems directly pass np array to attrs is not Allowed
    # Need to covert to list, and only support 1D (ONNX AttributeProto limit)
    block_h = 2
    block_w = 2
    attrs = {"input_format": ["NCHW_CUBE"],
    "output_format": ["YX_BLOCK"],
    "block_h": block_h,
    "block_w": block_w,
    "enable_parallel": True,
    "input_core_split": [16, 1], #core split factor = 16 at in channel axis = 1
    "output_core_split": [16, 2] #core split factor = 16 at out channel axis = 2
    # "real_in_shape": in_tensor.shape
    }
    # Note:
    # default return is a list, can also return a single output by [index]
    return self.layer(op="CustomOp",
        name=name,
        inputs=[in_tensor],
        # outputs=["output_unfold"],
        outputs=[out_tensor],
        attrs=attrs)

@gs.Graph.register()
def vastai_fold(self, in_tensor, out_tensor, name):
    # Note: seems directly pass np array to attrs is not Allowed
    # Need to covert to list, and only support 1D (ONNX AttributeProto limit)
    block_b = 2
    block_y = out_tensor.shape[3] // block_b
    attrs = {"input_format": ["YX_BLOCK"],
    "output_format": ["NCHW_CUBE"],
    "block_b": 2,
    "block_y": block_y,
    "enable_parallel": True,
    "input_core_split": [16, 2], #core split factor = 16 at in channel axis = 2
    "output_core_split": [16, 1] #core split factor = 16 at out channel axis = 1
    # "real_in_shape": in_tensor.shape
    }

    return self.layer(op="CustomOp",
        name=name,
        inputs=[in_tensor],
        # outputs=["output_fold"],
        outputs=[out_tensor],
        attrs=attrs)

def check_input(inputs, op_name):
    if len(inputs) != 0:
        if inputs[0].op == op_name:
            return True, inputs[0].inputs[0]
    return False, None

def find_pattern_in_out(op_pattern_list, op_input_dim_size, tmap):
    in_out_dict = {}
    for key, tensor_info in tmap.items():
        cur_tensor_info = tensor_info
        match = True
        for op_name in op_pattern_list:
            flag, pre_op_in_tensor = check_input(cur_tensor_info.inputs, op_name)
            if flag:
                cur_tensor_info = pre_op_in_tensor
            else:
                match = False
                break
        if match == True and len(cur_tensor_info.shape) == op_input_dim_size:
            print(f"pattern in: {cur_tensor_info.name}, pattern out: {key}, op_input_dim_size: {op_input_dim_size}")
            in_out_dict[cur_tensor_info.name] = key
    return in_out_dict


def fix_and_sim():
    graph = gs.import_onnx(onnx.load(onnx_model_path))


    tmap = graph.tensors()
    tmap[inputname].shape = (1, 3, 224, 224)
    graph.cleanup().toposort()

    for out in graph.outputs:
        out.dtype = np.float32
    onnx_sim_path  = onnx_model_path.replace(".onnx", "_sim.onnx")
    onnx.save(
        gs.export_onnx(graph),
        onnx_model_path.replace(".onnx", "_fixed.onnx")
    )

    print('start simplify ..')

    # onnx_fpath = "mobilevit-small_fix_zww.onnx"
    onnx_model = onnx.load_model(onnx_model_path.replace(".onnx", "_fixed.onnx"))
    model_simp, check = simplify(onnx_model)

    # Path(f'./sim/').mkdir(parents=True, exist_ok=True)
    onnx.save(model_simp, onnx_sim_path, save_as_external_data=True, all_tensors_to_one_file=True)
    assert check, "Simplified ONNX model could not be validated"
    print("onnxsim success.")
    return onnx_sim_path

def convert_custom_op(onnx_sim_path):
    graph = gs.import_onnx(onnx.load(onnx_sim_path))
    tmap = graph.tensors()
    unfold_op_list = ["Reshape", "Transpose", "Reshape", "Transpose", "Reshape"]
    fold_op_list = ["Reshape", "Transpose", "Reshape", "Transpose", "Reshape"]


    unfold_in_out_map = find_pattern_in_out(unfold_op_list, 4, tmap)
    fold_in_out_map = find_pattern_in_out(fold_op_list, 3, tmap)

    found = True
    if len(unfold_in_out_map) == 0:
        print("err: cant not find unfold op!")
        found = False
    if len(fold_in_out_map) == 0:
        print("err: cant not find fold op!")
        found = False
    if not found:
        exit()

    for in_name, out_name in unfold_in_out_map.items():
        tmap[in_name].outputs.clear()
        tmap[out_name].inputs.clear()
        name = f'VASTAI_OP_MVIT_UNFOLD'
        unfold = graph.vastai_unfold(tmap[in_name], tmap[out_name], name)

    for in_name, out_name in fold_in_out_map.items():
        tmap[in_name].outputs.clear()
        tmap[out_name].inputs.clear()
        name = f'VASTAI_OP_MVIT_FOLD'
        fold = graph.vastai_fold(tmap[in_name], tmap[out_name], name)

    graph.cleanup().toposort()

    for out in graph.outputs:
        out.dtype = np.float32
    onnx.save(gs.export_onnx(graph), onnx_sim_path.replace("sim.onnx", "sim_custom_op.onnx"))
    print("done")



if __name__ == "__main__":

    inputname = "input"
    onnx_model_path = "/home/wzp/code/modelzoo/0304/algorithm_modelzoo/mobilevit_s.onnx"
    onnx_sim_path = fix_and_sim()
    convert_custom_op(onnx_sim_path)
