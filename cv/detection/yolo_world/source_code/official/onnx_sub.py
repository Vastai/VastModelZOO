import onnx
import argparse


def get_sub_model_clip(onnx_path):
    # onnx_path = './code/model_check/vit_custom_models/weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6_new1_sim.onnx'
    node_start = ['input_ids', 'attention_mask']
    node_end = ['/baseModel/backbone/text_model/Div_output_0']# ['onnx::Reshape_2979']
    node_end = ['onnx::Reshape_2979']

    for i in range(len(node_end)):
        onnx_sub_path = onnx_path.replace('.onnx', '_sub.onnx')
        onnx.utils.extract_model(onnx_path, onnx_sub_path, node_start, node_end)

def get_batch_model_img(onnx_path):
    # onnx_path = './code/model_check/vit_custom_models/weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6_new1203_sim.onnx'
    node_start = ['images', 'onnx::Reshape_2979']
    # node_start = ['images', '/baseModel/backbone/text_model/Div_output_0']

    node_end = ['onnx::MatMul_3489', 'onnx::MatMul_3566', 'onnx::MatMul_3643', '3533', '3610', '3687']
    # node_end = ['/baseModel/head_module/cls_contrasts.0/Reshape_output_0',
                # '/baseModel/head_module/cls_contrasts.1/Reshape_output_0',
                # '/baseModel/head_module/cls_contrasts.2/Reshape_output_0',
                # '3268', '3329', '3390'
                # ]
                
    for i in range(len(node_end)):
        onnx_sub_path = onnx_path.replace('.onnx', '_sub.onnx')
        onnx.utils.extract_model(onnx_path, onnx_sub_path, node_start, node_end)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="onnx extract")
    parse.add_argument(
        "--onnx_file",
        type=str,
        default="onnx/images.onnx",
        help="path to save onnx files",
    )
    parse.add_argument(
        "--export",
        type=str,
        choices=["image", "text"],
        help="choose backbone to export"
    )
    args = parse.parse_args()

    if args.export == 'image':
        get_sub_model_clip(args.onnx_file)
    else:
        get_batch_model_img(args.onnx_file)