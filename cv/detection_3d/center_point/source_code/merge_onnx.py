#!/usr/bin/env python

import argparse
import copy
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper


def make_scatterND(model,
                   rpn_input_shape,
                   indices_shape,
                   mask_shape,
                   pfe_out_maxpool_name,
                   batch_size,
                   save_for_trt=False):
    output_shape = [
        batch_size, rpn_input_shape[2] * rpn_input_shape[3], rpn_input_shape[1]
    ]

    if save_for_trt:
        squeeze_node = helper.make_node(op_type='Squeeze', inputs=[pfe_out_maxpool_name], \
                                    outputs=['pfe_squeeze_1'], name='pfe_Squeeze_1', \
                                    axes = [3])
        transpose_node_1 = helper.make_node(op_type='Transpose', inputs=['pfe_squeeze_1',], \
                                        outputs=['pfe_transpose_1'], name='pfe_Transpose_1', \
                                        perm=[0,2,1])
        scatter_node = helper.make_node(op_type='ScatterND', inputs=['scatter_data', 'indices_input', 'pfe_transpose_1'], \
                                        outputs=['scatter_1'], name='ScatterND_1', output_shape=output_shape, index_shape=indices_shape)
        transpose_node_2 = helper.make_node(op_type='Transpose', inputs=['scatter_1',], \
                                            outputs=['pfe_transpose_2'], \
                                            name='pfe_Transpose_2', perm=[0,2,1])
        reshape_node = helper.make_node(op_type='Reshape', inputs=['pfe_transpose_2','pfe_reshape_shape'], \
                                        outputs=['rpn_input'], name='pfe_reshape_1')
    else:
        # transpose_node_1 = helper.make_node(op_type="Reshape", inputs=[pfe_out_maxpool_name,"pfe_reshape_shape"], \
        #                                 outputs=['pfe_transpose_1'], name="pfe_Transpose_1")
        scatter_node = helper.make_node(op_type='PointPillarScatterFunction', inputs=[pfe_out_maxpool_name, 'voxel_coords_xyz', 'mask'], \
                                        outputs=['spatial_features'], name='ScatterND_1', features = rpn_input_shape[1], sizex = rpn_input_shape[2], sizey = rpn_input_shape[3], sizez = 1)
        # reshape_node = helper.make_node(op_type="Reshape", inputs=["scatter_1","pfe_reshape_shape"], \
        #                                 outputs=['rpn_input'], name="pfe_reshape_1")

    squeeze_axes = [3]
    squeeze_tensor = np.array(squeeze_axes, dtype=np.int32)
    squeeze_tensor = numpy_helper.from_array(squeeze_tensor, name='axes')
    model.graph.initializer.append(squeeze_tensor)
    if save_for_trt:
        data_shape = [
            batch_size, rpn_input_shape[2] * rpn_input_shape[3],
            rpn_input_shape[1]
        ]
        data = np.zeros(data_shape, dtype=np.float32)
        data_tensor = numpy_helper.from_array(data, name='scatter_data')
        model.graph.initializer.append(data_tensor)
    else:
        data_shape = [
            batch_size, rpn_input_shape[1], rpn_input_shape[2],
            rpn_input_shape[3]
        ]
        data = np.zeros(data_shape, dtype=np.float32)
        data_tensor = numpy_helper.from_array(data, name='scatter_data')
        model.graph.initializer.append(data_tensor)

    # reshape_shape = np.array([1,-1,64], dtype=np.int64)
    # reshape_tensor = numpy_helper.from_array(reshape_shape, name="pfe_reshape_shape")
    # model.graph.initializer.append(reshape_tensor)

    if save_for_trt:
        idx_type = onnx.TensorProto.INT32
        input_node = onnx.helper.make_tensor_value_info(
            'indices_input', idx_type, indices_shape)
        model.graph.input.append(input_node)
    else:
        idx_type = onnx.TensorProto.INT16
        input_node = onnx.helper.make_tensor_value_info(
            'voxel_coords_xyz', idx_type, [indices_shape[1], indices_shape[2]])
        model.graph.input.append(input_node)
        mask_type = onnx.TensorProto.INT8
        mask_node = onnx.helper.make_tensor_value_info('mask', mask_type,
                                                       [1, mask_shape[1]])
        model.graph.input.append(mask_node)

    if save_for_trt:
        model.graph.node.append(squeeze_node)
        model.graph.node.append(transpose_node_2)
        model.graph.node.append(reshape_node)
        model.graph.node.append(transpose_node_1)

    model.graph.node.append(scatter_node)


def parse_int_list(value):
    return [int(x) for x in value.split(',')]


parser = argparse.ArgumentParser(description='RUN Det WITH VSX')
parser.add_argument('--pfe_model_path',
                    type=str,
                    default='./onnx_models/PillarVFE_32000_sim.onnx',
                    help='pfe model path')
parser.add_argument(
    '--rpn_model_path',
    type=str,
    default='./onnx_models/BaseBEVBackbone_CenterHead_sim.onnx',
    help='rpn model path')
parser.add_argument('--target_model_path',
                    type=str,
                    default='./onnx_models/',
                    help='target model path')
parser.add_argument('--target_model_name',
                    type=str,
                    default='model_centerpoint_pp_32000_v5.onnx',
                    help='target model path')
parser.add_argument('--max_voxel_num',
                    type=int,
                    default='32000',
                    help='max voxel num')
parser.add_argument('--backbone_input_shape',
                    type=parse_int_list,
                    default=[1, 64, 480, 480],
                    help='backbone input shape')
args = parser.parse_args()

if __name__ == '__main__':

    pfe_sim_model_path = args.pfe_model_path
    rpn_sim_model_path = args.rpn_model_path
    if not os.path.exists(args.target_model_path):
        os.makedirs(args.target_model_path)
    pointpillars_save_path = os.path.join(args.target_model_path,
                                          args.target_model_name)

    pfe_model = onnx.load(pfe_sim_model_path)
    rpn_model = onnx.load(rpn_sim_model_path)

    batch_size = 1
    points_num = args.max_voxel_num
    rpn_input_conv_name = '/backbone_2d/blocks.0/blocks.0.1/Conv'
    # pfe_out_maxpool_name = "47"
    pfe_out_Squeeze_name = 'pillar_features'
    rpn_input_shape = [
        batch_size, args.backbone_input_shape[1], args.backbone_input_shape[2],
        args.backbone_input_shape[3]
    ]
    indices_shape = [batch_size, 3, points_num]
    mask_shape = [batch_size, points_num]

    # for node in pfe_model.graph.node:
    #     node.name = "pfe_"+node.name

    pfe_model.graph.input[0].type.tensor_type.shape.dim[
        0].dim_value = points_num
    pfe_model.graph.output[0].type.tensor_type.shape.dim[
        0].dim_value = points_num
    for value_info in pfe_model.graph.value_info:
        value_info.type.tensor_type.shape.dim[0].dim_value = points_num

    # pfe_model_trt = copy.deepcopy(pfe_model)

    # Connect pfe and rpn with scatterND
    make_scatterND(pfe_model, rpn_input_shape, indices_shape, mask_shape,
                   pfe_out_Squeeze_name, batch_size)
    # make_scatterND(pfe_model_trt, rpn_input_shape, indices_shape, mask_shape, pfe_out_Squeeze_name, batch_size, save_for_trt=True)

    # merge nodes, outputs and initializers
    pfe_model.graph.node.extend(rpn_model.graph.node)
    pfe_model.graph.output.pop()
    pfe_model.graph.output.extend(rpn_model.graph.output)
    pfe_model.graph.initializer.extend(rpn_model.graph.initializer)

    def change_input(model):
        for node in model.graph.node:
            if node.name == rpn_input_conv_name:
                node.input[0] = 'spatial_features'
                break

            # model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = indices_shape[1]

    change_input(pfe_model)

    # change_input(pfe_model_trt)


    def change_input_node_name(model, input_names):
        for i, input in enumerate(model.graph.input):
            input_name = input_names[i]
            for node in model.graph.node:
                for i, name in enumerate(node.input):
                    if name == input.name:
                        node.input[i] = input_name
            input.name = input_name

    def change_output_node_name(model, output_names):
        for i, output in enumerate(model.graph.output):
            output_name = output_names[i]
            for node in model.graph.node:
                for i, name in enumerate(node.output):
                    if name == output.name:
                        node.output[i] = output_name
            output.name = output_name

    output_names = []
    # for i in range(6):
    #     output_names.append(f"reg_{i}")
    #     output_names.append(f"height_{i}")
    #     output_names.append(f"dim_{i}")
    #     output_names.append(f"rot_{i}")
    #     output_names.append(f"vel_{i}")
    #     output_names.append(f"hm_{i}")

    output_names.append(f'center')
    output_names.append(f'center_z')
    output_names.append(f'dim')
    output_names.append(f'rot')
    output_names.append(f'hm')

    change_output_node_name(pfe_model, output_names)
    # change_output_node_name(pfe_model_trt, output_names)

    onnx.save(pfe_model, pointpillars_save_path)
    # onnx.save(pfe_model_trt, pointpillars_trt_save_path)
    # print("pfe_model:",pfe_model.graph.initializer)

    print('Done')
