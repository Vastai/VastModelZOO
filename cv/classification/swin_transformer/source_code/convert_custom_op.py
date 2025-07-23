# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
'@Author :        melodylu
'@Email :   algorithm@vastaitech.com
'@Time  :     2025/07/23 14:58:14
'''


import onnx_graphsurgeon as gs
import onnx
import math

@gs.Graph.register()
def reshape(self, a, new_shape, name, output_name):
    output_tensor = gs.Variable(output_name, "float32", new_shape)
    return self.layer(op="Reshape", name=name, inputs=[a, new_shape], outputs=[output_tensor])

@gs.Graph.register()
def vastai_window_partition(self, input_tensor, window_size, H, W, C, output_name=""):
    attrs = {
                "window_size": window_size,
                "H": H,
                "W": W,
                "input_from_ddr": [False,],
                "enable_parallel": True,
                "input_core_split": [16,-1],
                "output_core_split": [16,-1],
                "outputs_dims": [2,],
                "input_format":["YX_BLOCK"],
                "output_format":["YX_BLOCK"],
            }
    output_tensor = gs.Variable(output_name, "float32", [(H // window_size) * (W // window_size), window_size * window_size, C])
    return self.layer(op="CustomOp", name="VASTAI_WINDOW_PARTITION_OP", inputs=[input_tensor],
                      outputs=[output_tensor],
                      attrs=attrs)

@gs.Graph.register()
def vastai_window_reverse(self, input_tensor, window_size, H, W, C, output_name=""):
    attrs = {
                "window_size": window_size,
                "H": H,
                "W": W,
                "input_from_ddr": [False,],
                "enable_parallel": True,
                "input_core_split": [16,-1],
                "output_core_split": [16,-1],
                "outputs_dims": [2,],
                "input_format":["YX_BLOCK"],
                "output_format":["YX_BLOCK"],
            }
    output_tensor = gs.Variable(output_name, "float32", [H * W, C])
    return self.layer(op="CustomOp", name="VASTAI_WINDOW_REVERSE_OP", inputs=[input_tensor],
                      outputs=[output_tensor],
                      attrs=attrs)

@gs.Graph.register()
def vastai_roll(self, input_tensor, H, W, C, shifts, output_name=""):
    attrs = {
                "shift_y": shifts[0],
                "shift_x": shifts[1],
                "height": H,
                "width": W,
                "dst_h_start": 0,
                "dst_h_end": H,
                "input_from_ddr": [False,],
                "enable_parallel": True,
                "input_core_split": [16,-1],
                "output_core_split": [16,-1],
                "outputs_dims": [2,],
                "input_format":["YX_BLOCK"],
                "output_format":["YX_BLOCK"],
            }
    output_tensor = gs.Variable(output_name, "float32", [H * W, C])
    return self.layer(op="CustomOp", name="VASTAI_ROLL_OP", inputs=[input_tensor],
                      outputs=[output_tensor],
                      attrs=attrs)

@gs.Graph.register()
def vastai_patch_merging(self, input_tensor, H, W, C, output_name=""):
    attrs = {
                "h": H*2,
                "w": W*2,
                "input_from_ddr": [False,],
                "enable_parallel": False,
                "input_core_split": [16,-1],
                "output_core_split": [16,-1],
                "outputs_dims": [2,],
                "input_format":["YX_BLOCK"],
                "output_format":["YX_BLOCK"],
            }
    output_tensor = gs.Variable(output_name, "float32", [H*W, C])
    return self.layer(op="CustomOp", name="VASTAI_PATCH_MERGING_OP", inputs=[input_tensor],
                      outputs=[output_tensor],
                      attrs=attrs)



def check_input(inputs, op_name):
    if len(inputs) != 0:
        if inputs[0].op == op_name:
            return True, inputs[0].inputs[0]
    return False, None


def convert_window_partition(graph):
    window_partition_op_list = ["Reshape", "Transpose", "Reshape"]
    tmap = graph.tensors()

    in_out_map = {}
    window_size = 0
    H = 0
    W = 0
    C = 0
    attrs = {}
    for key, tensor_info in tmap.items():
        cur_tensor_info = tensor_info
        match = True
        count = 0
        for op_name in window_partition_op_list:
            flag, pre_op_in_tensor = check_input(cur_tensor_info.inputs, op_name)
            if flag:
                if count == 0 and len(cur_tensor_info.shape) == 3:
                    window_size = int(math.sqrt(cur_tensor_info.shape[-2]))
                    if window_size*window_size == cur_tensor_info.shape[-2] and window_size > 1:
                        match = True
                    else:
                        match = False
                        break
                elif count == 1 and len(cur_tensor_info.shape) == 6:
                    if cur_tensor_info.shape[-2] == window_size and cur_tensor_info.shape[-3] == window_size:
                        match = True
                    else:
                        match = False
                        break
                elif count == 2 and len(cur_tensor_info.shape) == 6:
                    if cur_tensor_info.shape[-2] == window_size and cur_tensor_info.shape[-4] == window_size:
                        match = True
                        H = cur_tensor_info.shape[-2] * cur_tensor_info.shape[-3]
                        W = cur_tensor_info.shape[-4] * cur_tensor_info.shape[-5]
                        C = cur_tensor_info.shape[-1]
                    else:
                        match = False
                        break
                else:
                    match = False
                    break

                cur_tensor_info = pre_op_in_tensor
            else:
                match = False
                break
            count = count + 1
        if match:
            in_out_map[cur_tensor_info.name] = key
            attrs[cur_tensor_info.name] = [window_size, H, W, C]
            # break

    window_partition_custom_op_map = {}
    wp_out_name = ""
    for node in graph.nodes:
        # print(node.inputs[0].name)
        if node.outputs[0].name in in_out_map.keys():
            window_size, H, W, C = attrs[node.outputs[0].name]
            wp_out_name = in_out_map[node.outputs[0].name]

            if H == window_size and W == window_size:
                window_partition_custom_op_map[wp_out_name] = node.outputs[0]
            else:
                reshape_op = graph.reshape(node.outputs[0], [1*H*W, C], f"{node.outputs[0].name}_reshape", f"{node.outputs[0].name}_reshape_0")
                custom_op = graph.vastai_window_partition(reshape_op[0], window_size=window_size, H=H, W=W, C=C, output_name=f"onnx::vastai_window_partition_{len(window_partition_custom_op_map)}")
                window_partition_custom_op_map[wp_out_name] = custom_op[0]
        else:
            for i in range(len(node.inputs)):
                if node.inputs[i].name in window_partition_custom_op_map.keys():
                    node.inputs[i] = window_partition_custom_op_map[node.inputs[i].name]
                    break
    graph.cleanup().toposort()
    return graph

def convert_window_reverse(graph):

    tmap = graph.tensors()

    op_list = ["Reshape", "Transpose", "Reshape"]
    tmap = graph.tensors()

    in_out_map = {}
    window_size = 0
    H = 0
    W = 0
    C = 0
    attrs = {}
    for key, tensor_info in tmap.items():
        cur_tensor_info = tensor_info
        match = True
        count = 0
        for op_name in op_list:
            flag, pre_op_in_tensor = check_input(cur_tensor_info.inputs, op_name)
            if flag:
                if count == 0 and len(cur_tensor_info.inputs[0].inputs[0].shape) == 6:
                    if 1 == cur_tensor_info.shape[0]:
                        match = True
                        window_size = cur_tensor_info.inputs[0].inputs[0].shape[-2]
                        W = cur_tensor_info.inputs[0].inputs[0].shape[-2] * cur_tensor_info.inputs[0].inputs[0].shape[-3]
                        H = cur_tensor_info.inputs[0].inputs[0].shape[-4] * cur_tensor_info.inputs[0].inputs[0].shape[-5]
                        C = cur_tensor_info.inputs[0].inputs[0].shape[-1]
                    else:
                        match = False
                        break
                elif count == 1 and len(cur_tensor_info.shape) == 6:
                    if cur_tensor_info.shape[-2] == cur_tensor_info.shape[-4] and cur_tensor_info.shape[-2] > 1 and window_size == cur_tensor_info.shape[-2]:
                        match = True
                    else:
                        match = False
                        break
                elif count == 2 and len(cur_tensor_info.shape) == 6:
                    if cur_tensor_info.shape[-2] == window_size and cur_tensor_info.shape[-3] == window_size :
                        match = True
                    else:
                        match = False
                        break
                else:
                    match = False
                    break

                cur_tensor_info = pre_op_in_tensor
            else:
                match = False
                break
            count = count + 1
        if match:
            in_out_map[cur_tensor_info.name] = key
            attrs[cur_tensor_info.name] = [window_size, H, W, C]
            # break

    custom_op_map = {}
    wr_out_name = ""
    for node in graph.nodes:
        if node.outputs[0].name in in_out_map.keys():
            wr_out_name = in_out_map[node.outputs[0].name]
            window_size, H, W, C = attrs[node.outputs[0].name]

            if H == window_size and W == window_size:
                custom_op_map[wr_out_name] = node.outputs[0]
            else:
                reshape_op0 = graph.reshape(node.outputs[0], [(H // window_size) * (W // window_size), window_size * window_size, C], f"{node.outputs[0].name}_reshape", f"{node.outputs[0].name}_reshape_0")
                custom_op = graph.vastai_window_reverse(reshape_op0[0], window_size=window_size, H=H, W=W, C=C, output_name=f"onnx::vastai_window_reverse_{len(custom_op_map)}")
                custom_op_map[wr_out_name] = custom_op[0]
        else:
            for i in range(len(node.inputs)):
                if node.inputs[i].name in custom_op_map.keys():
                    reshape_op1 = graph.reshape(custom_op_map[node.inputs[i].name], node.inputs[0].shape, f"{node.outputs[0].name}_reshape", f"{node.outputs[0].name}_reshape_1")
                    node.inputs[i] = reshape_op1[0]
                    wr_out_name = ""
                    break

    graph.cleanup().toposort()
    return graph

def convert_roll(graph):
    tmap = graph.tensors()
    op_list = ["Concat", "Slice", "Concat", "Slice"]

    in_out_map = {}
    shift_y = 0
    shift_x = 0
    H = 0
    W = 0
    C = 0
    attrs = {}
    for key, tensor_info in tmap.items():
        cur_tensor_info = tensor_info
        match = True
        count = 0
        for op_name in op_list:
            flag, pre_op_in_tensor = check_input(cur_tensor_info.inputs, op_name)
            if flag:
                if count == 0 and len(cur_tensor_info.shape) == 4:
                    if 1 == cur_tensor_info.shape[0] and cur_tensor_info.shape[1] == cur_tensor_info.shape[2] and cur_tensor_info.shape[1] > 1:
                        match = True
                        H = cur_tensor_info.shape[1]
                        W = cur_tensor_info.shape[2]
                        C = cur_tensor_info.shape[3]
                    else:
                        match = False
                        break
                elif count == 1 and len(cur_tensor_info.shape) == 4:
                    if cur_tensor_info.shape[-3] == H:
                        match = True

                        shift_x = -cur_tensor_info.inputs[0].inputs[1].values[0]
                    else:
                        match = False
                        break
                elif count == 2 and len(cur_tensor_info.shape) == 4:
                    if cur_tensor_info.shape[-2] == W and cur_tensor_info.shape[-3] == H :
                        match = True
                    else:
                        match = False
                        break
                elif count == 3 and len(cur_tensor_info.shape) == 4:
                    if cur_tensor_info.shape[-2] == W:
                        match = True
                        shift_y = -cur_tensor_info.inputs[0].inputs[1].values[0]
                    else:
                        match = False
                        break
                else:
                    match = False
                    break

                cur_tensor_info = pre_op_in_tensor
            else:
                match = False
                break
            count = count + 1
        if match:
            in_out_map[cur_tensor_info.name] = key
            attrs[cur_tensor_info.name] = [shift_y, shift_x, H, W, C]
            # break
    custom_op_map = {}
    wr_out_name = ""
    for node in graph.nodes:
        # print(node.inputs[0].name)
        if node.outputs[0].name in in_out_map.keys():
            wr_out_name = in_out_map[node.outputs[0].name]
            shift_y, shift_x, H, W, C = attrs[node.outputs[0].name]

            reshape_op0 = graph.reshape(node.outputs[0], [H*W, C], f"{node.outputs[0].name}_reshape", f"{node.outputs[0].name}_reshape_0")
            custom_op = graph.vastai_roll(reshape_op0[0], H=H, W=W, C=C, shifts=[shift_y, shift_x], output_name=f"onnx::vastai_rool_{len(custom_op_map)}")
            custom_op_map[wr_out_name] = custom_op[0]
        else:
            for i in range(len(node.inputs)):
                if node.inputs[i].name in custom_op_map.keys():
                    reshape_op1 = graph.reshape(custom_op_map[node.inputs[i].name], node.inputs[0].shape, f"{node.outputs[0].name}_reshape", f"{node.outputs[0].name}_reshape_1")
                    node.inputs[i] = reshape_op1[0]
                    wr_out_name = ""
                    break
    graph.cleanup().toposort()
    return graph


def convert_patch_merging(graph):
    tmap = graph.tensors()
    op_list = ["Concat", "Slice", "Reshape"]

    in_out_map = {}
    H = 0
    W = 0
    C = 0
    attrs = {}
    for key, tensor_info in tmap.items():
        cur_tensor_info = tensor_info
        match = True
        count = 0
        for op_name in op_list:
            flag, pre_op_in_tensor = check_input(cur_tensor_info.inputs, op_name)
            if flag:
                if count == 0 and len(cur_tensor_info.shape) == 4:
                    if 1 == cur_tensor_info.shape[0] and cur_tensor_info.shape[1] == cur_tensor_info.shape[2] and cur_tensor_info.shape[1] > 1:
                        match = True
                        H = cur_tensor_info.shape[1]
                        W = cur_tensor_info.shape[2]
                        C = cur_tensor_info.shape[3]
                    else:
                        match = False
                        break
                elif count == 1 and len(cur_tensor_info.shape) == 4 and len(cur_tensor_info.inputs[0].inputs) == 5 and len(cur_tensor_info.inputs[0].inputs[4].values) == 2 and cur_tensor_info.inputs[0].inputs[4].values[0] == 2:
                    if cur_tensor_info.shape[-3] == H:
                        match = True
                    else:
                        match = False
                        break
                elif count == 2 and len(cur_tensor_info.shape) == 4:
                    if cur_tensor_info.shape[-2] == W*2 and cur_tensor_info.shape[-3] == H*2 and  cur_tensor_info.shape[-1] == C // 4:
                        match = True
                    else:
                        match = False
                        break
                else:
                    match = False
                    break

                cur_tensor_info = pre_op_in_tensor
            else:
                match = False
                break
            count = count + 1
        if match:
            in_out_map[cur_tensor_info.name] = key
            attrs[cur_tensor_info.name] = [H, W, C]
            # break
    custom_op_map = {}
    wr_out_name = ""
    for node in graph.nodes:
        # print(node.inputs[0].name)
        if node.outputs[0].name in in_out_map.keys():
            wr_out_name = in_out_map[node.outputs[0].name]
            H, W, C = attrs[node.outputs[0].name]

            reshape_op0 = graph.reshape(node.outputs[0], [4*H*W, C // 4], f"{node.outputs[0].name}_reshape", f"{node.outputs[0].name}_reshape_0")
            custom_op = graph.vastai_patch_merging(reshape_op0[0], H=H, W=W, C=C, output_name=f"onnx::vastai_patch_merge_{len(custom_op_map)}")
            custom_op_map[wr_out_name] = custom_op[0]
        else:
            for i in range(len(node.inputs)):
                if node.inputs[i].name in custom_op_map.keys():
                    reshape_op1 = graph.reshape(custom_op_map[node.inputs[i].name], node.inputs[0].shape, f"{node.outputs[0].name}_reshape", f"{node.outputs[0].name}_reshape_1")
                    node.inputs[i] = reshape_op1[0]
                    wr_out_name = ""
                    break
    graph.cleanup().toposort()
    return graph

def rewrite_swin_base_patch4_window7_224_onnx():

    graph = gs.import_onnx(onnx.load(onnx_path))
    graph = convert_window_partition(graph)
    graph = convert_window_reverse(graph)
    graph = convert_roll(graph)
    graph = convert_patch_merging(graph)

    onnx.save(
        gs.export_onnx(graph),
        onnx_op_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True
    )
    print("success")





if __name__ == '__main__':
    onnx_path = "./swin_base_patch4_window7_224_sim.onnx"
    onnx_op_path = onnx_path.replace("sim.onnx", "sim_custom_op.onnx")
    rewrite_swin_base_patch4_window7_224_onnx()

