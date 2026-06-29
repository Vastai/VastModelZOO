import os
import tqdm
import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort


def get_image_data(image_file, input_shape = [1, 3, 80, 160]):
    """Load image, resize, normalize, convert to NCHW format."""
    size = [input_shape[2], input_shape[3]] # h,w

    # https://github.com/PaddlePaddle/PaddleX/blob/release/3.6/paddlex/inference/models/image_classification/predictor.py#L155
    mean = [
        0.485,
        0.456,
        0.406
    ]
    std = [
        0.229,
        0.224,
        0.225
    ]

    image = Image.open(image_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)

    image = np.ascontiguousarray(image)
    if mean[0] < 1 and std[0] < 1:
        image = image.astype(np.float32)
        image /= 255.0
        image -= np.array(mean)
        image /= np.array(std)
    else:
        image = image - np.array(mean)  # mean
        image /= np.array(std)  # std
    image = image.transpose((2, 0, 1))  # HWC -> CHW
    image = image[np.newaxis, :]  # CHW -> NCHW
    return image


def parse_int_list(value):
    return [int(x) for x in value.split(',')]


parser = argparse.ArgumentParser(description="PP-LCNet doc orientation classification — ONNX accuracy benchmark")
parser.add_argument("--image_dir", type=str, default="datasets/textline_orientation_example_data", help="Dataset root (joined with relative paths in label_file)")
parser.add_argument("--model_input_shape", type=parse_int_list, default=[1, 3, 80, 160], help="model input shape NCHW")
parser.add_argument("--onnx_model_path", type=str, default="weights/PP-LCNet_x1_0_textline_ori_infer_inference_sim.onnx", help="ONNX model file path; skip ONNX evaluation if empty")
parser.add_argument("--output_file", type=str, default="0cls_pred.txt", help="Output file for predictions")
parser.add_argument("--label_file", type=str, default="datasets/textline_orientation_example_data/val.txt", help="Label file (format: <relative_path> <label_index>) for accuracy evaluation")
parser.add_argument("--num_images", type=int, default=-1, help="Number of images to test; -1 means all")
args = parser.parse_args()


if __name__ == '__main__':
    # ── Read image list and labels from label_file ──
    images = []
    labels_dict = {}
    label_list = ["0", "180"]
    
    with open(args.label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_rel, gt_label = parts[0], int(parts[1])
            images.append(os.path.join(args.image_dir, img_rel))
            labels_dict[os.path.basename(img_rel)] = gt_label


    # ── Load ONNX model ──
    use_onnx = False
    if args.onnx_model_path and os.path.exists(args.onnx_model_path):
        use_onnx = True
        providers = ["CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        onnx_model = ort.InferenceSession(args.onnx_model_path, sess_opts, providers=providers)
        print(f"ONNX model loaded: {args.onnx_model_path}")
    else:
        print("ONNX model path not set or file not found, skip ONNX evaluation")

    if not use_onnx:
        print("Error: no models available, exit")
        exit(1)

    # ── Limit number of images ──
    if args.num_images > 0 and args.num_images < len(images):
        images = images[:args.num_images]

    correct, total = 0, 0

    with open(args.output_file, "w") as outfile:
        for img_path in tqdm.tqdm(images, desc="Evaluating"):
            if not os.path.exists(img_path):
                print(f"Warning: image not found, skipping: {img_path}")
                continue
            basename = os.path.splitext(os.path.basename(img_path))[0]
            gt_label = labels_dict.get(os.path.basename(img_path), -1)
            if gt_label < 0:
                continue
            total += 1

            # resize
            img = get_image_data(img_path, input_shape=args.model_input_shape)

            parts = []
            if use_onnx:
                inp_name = onnx_model.get_inputs()[0].name
                logits = onnx_model.run(None, {inp_name: img})[0]
                assert isinstance(logits, np.ndarray)
                onnx_idx = logits.argmax(axis=1)[0]
                onnx_out = (label_list[onnx_idx], float(logits[0, onnx_idx]))
                parts.append(f"ONNX={onnx_out}")
                if onnx_idx == gt_label:
                    correct += 1

            print(f"{img_path}: {', '.join(parts)}")
            outfile.write(f"{img_path} {' '.join(parts)}\n")

    # ── Compute Top-1 accuracy ──
    if total > 0:
        acc = correct / total * 100.0
        print("\n========== Top-1 Accuracy ==========")
        print(f"ONNX  Top-1: {acc:.2f}%")
        print(f"Total : {total}")
        print("====================================\n")



'''
此处测试精度基于: https://paddle-model-ecology.bj.bcebos.com/paddlex/data/textline_orientation_example_data.tar

h80 w160

PP-LCNet_x0_25_textline_ori_infer_inference_sim.onnx
========== Top-1 Accuracy ==========
ONNX  Top-1: 91.00%
Total : 200
====================================

PP-LCNet_x1_0_textline_ori_infer_inference_sim.onnx
========== Top-1 Accuracy ==========
ONNX  Top-1: 85.50%
Total  : 200
====================================
'''
