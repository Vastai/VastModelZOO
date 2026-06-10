import os
import tqdm
import argparse
import numpy as np

# from paddleocr import DocImgOrientationClassification
from paddleocr import TextLineOrientationClassification

parser = argparse.ArgumentParser(description="PP-LCNet doc orientation — PaddleOCR accuracy benchmark")
parser.add_argument("--file_path", type=str, default="datasets/textline_orientation_example_data",
                    help="Dataset root (joined with relative paths in label_file)")
parser.add_argument("--model_name", type=str, default="PP-LCNet_x0_25_textline_ori",
                    help="Pretrained model name (e.g. PP-LCNet_x0_25_textline_ori)")
parser.add_argument("--model_dir", type=str, default="",
                    help="Local model directory (takes precedence over model_name)")
parser.add_argument("--output_file", type=str, default="pp_pred.txt", help="Output file for predictions")
parser.add_argument("--label_file", type=str, default="datasets/textline_orientation_example_data/val.txt",
                    help="Label file (format: <relative_path> <label_index>) for accuracy evaluation")
parser.add_argument("--num_images", type=int, default=-1,
                    help="Number of images to test; -1 means all")
args = parser.parse_args()

if __name__ == '__main__':
    # ── Read image list and labels from label_file ──
    images = []
    labels_dict = {}
    # label_list = ["0", "180"]

    with open(args.label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_rel, gt_label = parts[0], int(parts[1])
            images.append(os.path.join(args.file_path, img_rel))
            labels_dict[os.path.basename(img_rel)] = gt_label

    # ── Limit number of images ──
    if args.num_images > 0 and args.num_images < len(images):
        images = images[:args.num_images]

    # ── Load PaddleOCR model ──
    kwargs = {}
    if args.model_dir:
        kwargs["model_dir"] = args.model_dir
    else:
        kwargs["model_name"] = args.model_name
    model = TextLineOrientationClassification(**kwargs)

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

            # predict with topk=1
            output = model.predict(img_path, topk=1)
            res = list(output)[0]
            pred_idx = int(res["class_ids"][0])
            pred_label = res["label_names"][0]
            score = float(res["scores"][0])

            if pred_idx == gt_label:
                correct += 1

            decode_out = (pred_label, score)
            print(f"{img_path}: Paddle={decode_out}")
            outfile.write(f"{img_path} Paddle={decode_out}\n")

    # ── Compute Top-1 accuracy ──
    if total > 0:
        acc = correct / total * 100.0
        print("\n========== Top-1 Accuracy ==========")
        print(f"Paddle Top-1: {acc:.2f}%")
        print(f"Total : {total}")
        print("====================================\n")


'''
此处测试精度基于: https://paddle-model-ecology.bj.bcebos.com/paddlex/data/textline_orientation_example_data.tar

h80 w160
尺寸信息来自原始权重配置: PP-LCNet_x0_25_textline_ori_infer/inference.yml

PP-LCNet_x0_25_textline_ori
========== Top-1 Accuracy ==========
Paddle Top-1: 89.50%
Total : 200
====================================

PP-LCNet_x1_0_textline_ori
========== Top-1 Accuracy ==========
Paddle Top-1: 87.00%
Total : 200
====================================
'''