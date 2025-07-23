
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

from groundingdino import GroundingDino
import vaststreamx as vsx
import cv2
import argparse
from tqdm import tqdm
import numpy as np

def load_labels(file):
    with open(file) as fin:
        return [line.strip() for line in fin.readlines()]

def cv_bgr888_to_nv12(bgr888):
    yuv_image = cv2.cvtColor(bgr888, cv2.COLOR_BGR2YUV_I420)
    height, width = bgr888.shape[:2]
    y = yuv_image[:height, :]
    u = yuv_image[height : height + height // 4, :]
    v = yuv_image[height + height // 4 :, :]
    u = np.reshape(u, (height // 2, width // 2))
    v = np.reshape(v, (height // 2, width // 2))
    uv_plane = np.empty((height // 2, width), dtype=np.uint8)
    uv_plane[:, 0::2] = u
    uv_plane[:, 1::2] = v
    yuv_nv12 = np.concatenate((y, uv_plane), axis=0)
    return yuv_nv12

def cv_bgr888_to_vsximage(bgr888, vsx_format, device_id):
    h, w = bgr888.shape[:2]
    if vsx_format == vsx.ImageFormat.BGR_INTERLEAVE:
        res = bgr888
    elif vsx_format == vsx.ImageFormat.BGR_PLANAR:
        res = np.array(bgr888).transpose(2, 0, 1)
    elif vsx_format == vsx.ImageFormat.RGB_INTERLEAVE:
        res = cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)
    elif vsx_format == vsx.ImageFormat.RGB_PLANAR:
        res = np.array(cv2.cvtColor(bgr888, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    elif vsx_format == vsx.ImageFormat.YUV_NV12:
        res = cv_bgr888_to_nv12(bgr888=bgr888)
    else:
        assert False, f"Unsupport format:{vsx_format}"
    return vsx.create_image(
        res,
        vsx_format,
        w,
        h,
        device_id,
    )

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgmod_prefix",
        default="/path/to/image_encode_model/mod",
        help="image model prefix of the model suite files",
    )
    parser.add_argument(
        "--imgmod_hw_config",
        help="image model hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--imgmod_vdsp_params",
        default="../vacc_code/vdsp_params/image_encoder-vdsp_params.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--txtmod_prefix",
        default="/path/to/text_encoder/mod",
        help="text model prefix of the model suite files",
    )
    parser.add_argument(
        "--txtmod_vdsp_params",
        default="../vacc_code/vdsp_params/text_encoder-vdsp_params.json",
        help="text model vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--txtmod_hw_config",
        help="text model hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--decmod_prefix",
        default="/path/to/clip_txt_vdsp.json",
        help="text model vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--decmod_hw_config",
        help="text model hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="/path/to/tokenizer/clip-vit-base-patch32",
        help="tokenizer path",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )

    parser.add_argument(
        "--threshold",
        default=0.25,
        type=float,
        help="object confidence threshold",
    )
    parser.add_argument(
        "--label_file",
        default="/path/to/coco2id.txt",
        help="label file",
    )
    parser.add_argument(
        "--input_file",
        default="/path/to/dog.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="",
        help="output file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="input dataset filelist",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="input dataset root",
    )
    parser.add_argument(
        "--dataset_output_folder",
        default="",
        help="dataset output folder path",
    )
    args = parser.parse_args()
    return args


# caption =  'person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . traffic light . fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . broccoli . carrot . hot dog . pizza . donut . cake . chair . couch . potted plant . bed . dining table . toilet . tv . laptop . mouse . remote . keyboard . cell phone . microwave . oven . toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . hair drier . toothbrush .'

if __name__ == "__main__":
    args = argument_parser()
    assert vsx.set_device(args.device_id) == 0
    labels = load_labels(args.label_file)
    batch_size = 1

    caption = " . ".join(labels) + " ."

    class_dict = {}

    for index, label in enumerate(labels):
        class_dict[label] = index

    model = GroundingDino(
        args.txtmod_prefix,
        args.txtmod_vdsp_params,
        args.imgmod_prefix,
        args.imgmod_vdsp_params,
        args.decmod_prefix,
        args.tokenizer_path,
        args.label_file,
        batch_size,
        args.device_id,
        args.threshold,
        args.txtmod_hw_config,
        args.imgmod_hw_config,
        args.decmod_hw_config,
    )

    image_format = model.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"failed to open: {args.input_file}"
        vsx_image = cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        txt_out = model.process_text(caption)
        outputs = model.process_image_and_decode(txt_out, vsx_image)
        color = [0, 0, 255]
        print("Detection objects:")
        for output in outputs:
            score, box, class_name = output[:3]
            print(
                f"Object class: {class_name}, score: {score:.6f}, bbox:[",
                ",".join(f"{num:.2f}" for num in box),
                "]",
            )
            if args.output_file != "":
                box = box.astype(int)
                cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), color)
                font = cv2.FONT_HERSHEY_SIMPLEX
                left, top = box[:2]
                top += 15
                text = f"{class_name},{score:.2f}"
                cv2.putText(cv_image, text, (left, top), font, 0.5, color)
        if args.output_file != "":
            cv2.imwrite(args.output_file, cv_image)
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]
        txt_out = model.process_text(caption)
        for image_file in tqdm(filelist):
            fullname = os.path.join(args.dataset_root, image_file)
            cv_image = cv2.imread(fullname)
            assert cv_image is not None, f"Failed to read {fullname}"
            vsx_image = cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            outputs = model.process_image_and_decode(txt_out, vsx_image)
            base_name, _ = os.path.splitext(os.path.basename(image_file))
            outfile = open(
                os.path.join(args.dataset_output_folder, base_name + ".txt"), "wt"
            )
            print(f"{image_file} detection objects:")
            for output in outputs:
                score, bbox, class_name = output[:3]
                print(f"Object class: {class_name}, score: {score}, bbox: {bbox}")
                outfile.write(
                    f"{class_name} {score} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                )
            outfile.close()
