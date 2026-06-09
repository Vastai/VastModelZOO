import os
import time
import json
import argparse
from queue import Queue
from threading import Event, Thread

import cv2
import tqdm
import numpy as np

import vaststreamx as vsx


def topk_accuracy(scores, labels, topk=(1,)):
    """Compute top-k accuracy (percentage)."""
    maxk = max(topk)
    batch_size = len(labels)
    pred_topk = np.argsort(scores, axis=1)[:, -maxk:][:, ::-1]

    correct = np.zeros((batch_size, maxk), dtype=bool)
    for i in range(batch_size):
        for j in range(maxk):
            if pred_topk[i, j] == labels[i]:
                correct[i, j:] = True
                break

    res = {}
    for k in topk:
        correct_k = correct[:, :k].any(axis=1).sum()
        res[k] = correct_k * 100.0 / batch_size
    return res


def cv_bgr888_to_nv12(bgr888):
    yuv_image = cv2.cvtColor(bgr888, cv2.COLOR_BGR2YUV_I420)
    height, width = bgr888.shape[:2]
    y = yuv_image[:height, :]
    u = yuv_image[height: height + height // 4, :]
    v = yuv_image[height + height // 4:, :]
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
        raise ValueError(f"Unsupported format: {vsx_format}")
    return vsx.create_image(res, vsx_format, w, h, device_id)


class Classification:
    def __init__(self,
                 model_prefix_path,
                 vdsp_params_info,
                 device_id=0,
                 batch_size=1,
                 balance_mode=0,
                 is_async_infer=False,
                 model_output_op_name=""):
        if isinstance(vdsp_params_info, str):
            with open(vdsp_params_info) as f:
                json.load(f)

        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.preprocess_name = "preprocess_res"
        self.input_id = 0
        self.balance_mode = {0: vsx.StreamBalanceMode.ONCE, 1: vsx.StreamBalanceMode.RUN}

        self.attr = vsx.AttrKey
        assert vsx.set_device(self.device_id) == 0

        self.model = vsx.Model(model_prefix_path, batch_size)
        self.fusion_op = vsx.Operator.load_ops_from_json_file(vdsp_params_info)[0]

        self.graph = vsx.Graph()
        self.model_op = vsx.ModelOperator(self.model)
        self.graph.add_operators(self.fusion_op, self.model_op)

        self.infer_stream = vsx.Stream(self.graph, self.balance_mode[balance_mode])
        if is_async_infer and len(self.model_output_op_name) > 0:
            self.infer_stream.register_operator_output(self.model_output_op_name, self.model_op)
        else:
            self.infer_stream.register_operator_output(self.model_op)

        n, c, h, w = self.model.input_shape[0]
        self.infer_stream.register_operator_output(
            self.preprocess_name, self.fusion_op, [[(c, h, w), vsx.TypeFlag.FLOAT16]])

        self.infer_stream.build()
        self.current_id = -1
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}

        self.consumer = Thread(target=self.async_receive_infer)
        self.consumer.start()

    def async_receive_infer(self):
        while True:
            try:
                if len(self.model_output_op_name) > 0:
                    result = self.infer_stream.get_operator_output(self.model_output_op_name)
                else:
                    result = self.infer_stream.get_operator_output(self.model_op)
                if result is not None:
                    self.current_id += 1
                    input_id, height, width = self.input_dict[self.current_id]
                    model_output_list = [[vsx.as_numpy(out).astype(np.float32) for out in result[0]]]
                    self.result_dict[input_id] = []
                    self.post_processing(input_id, height, width, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                print(str(e))
                break

    def finish(self):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def post_processing(self, input_id, height, width, stream_output_list):
        output_data = stream_output_list[0][0]
        self.result_dict[input_id].append({"features": output_data})

    def calculate_padding(self, model_width, model_height, image_width, image_height):
        ratio = image_width / image_height
        resize_h = model_height
        if model_height * ratio > model_width:
            resize_w = model_width
        else:
            resize_w = int(model_height * ratio)
        right = model_width - resize_w if model_width - resize_w > 0 else 0
        return (int(resize_w), resize_h, 0, 0, 0, right)

    def _run(self, image):
        cv_image = cv2.imread(image)
        assert cv_image is not None, f"Failed to read: {image}"
        vsx_image = cv_bgr888_to_vsximage(cv_image, vsx.ImageFormat.RGB_PLANAR, self.device_id)
        width = cv_image.shape[1]
        height = cv_image.shape[0]
        ext_op_config = self.calculate_padding(
            self.model.input_shape[0][3], self.model.input_shape[0][2], width, height)

        input_id = self.input_id
        self.input_dict[input_id] = (input_id, height, width)
        self.event_dict[input_id] = Event()

        self.infer_stream.run_async([vsx_image], {
            "rgb_letterbox_ext": [ext_op_config]
        })

        self.input_id += 1
        return input_id

    def run(self, image):
        input_id = self._run(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]
        return result

    def run_batch(self, images):
        queue = Queue(20)

        def input_thread():
            for image in tqdm.tqdm(images):
                input_id = self._run(image)
                queue.put(input_id)
            queue.put(None)

        thread = Thread(target=input_thread)
        thread.start()
        while True:
            input_id = queue.get()
            if input_id is None:
                break
            self.event_dict[input_id].wait()
            result = self.result_dict.pop(input_id)
            del self.event_dict[input_id]
            yield result

    def run_sync(self, image):
        cv_image = cv2.imread(image)
        assert cv_image is not None, f"Failed to read: {image}"
        vsx_image = cv_bgr888_to_vsximage(cv_image, vsx.ImageFormat.RGB_PLANAR, self.device_id)
        width = cv_image.shape[1]
        height = cv_image.shape[0]
        ext_op_config = self.calculate_padding(
            self.model.input_shape[0][3], self.model.input_shape[0][2], width, height)
        output = self.infer_stream.run_sync([vsx_image], {
            "rgb_letterbox_ext": [ext_op_config]
        })
        model_output_list = [[vsx.as_numpy(out).astype(np.float32) for out in output[0]]]
        output_data = model_output_list[0][0]
        return [{"features": output_data}]


parser = argparse.ArgumentParser(description="PP-LCNet doc orientation classification — VSX accuracy benchmark")
parser.add_argument("--file_path", type=str, default="datasets/text_image_orientation",
                    help="Dataset root (joined with relative paths in label_file)")
parser.add_argument("--label_file", type=str, default="datasets/text_image_orientation/val.txt",
                    help="Label file (format: <relative_path> <label_index>) for accuracy evaluation")
parser.add_argument("--num_images", type=int, default=-1,
                    help="Number of images to test; -1 means all")
parser.add_argument("--model_prefix_path", type=str, default="deploy_weights/PP-LCNet_x1_0_doc_ori_infer_inference_sim_vacc/mod",
                    help="VSX model prefix path")
parser.add_argument("--vdsp_params_info", type=str, default="cv/classification/pplcnet_doc_ori/build_in/vdsp_params/pplcnet_x1_0_doc_ori-vdsp_params-resize.json",
                    help="VS DSP operator JSON config")
parser.add_argument("--output_file", type=str, default="vsx_pred.txt", help="Output file for predictions")
parser.add_argument("--device_id", type=int, default=0, help="VSX device ID")
parser.add_argument("--batch", type=int, default=1, help="Batch size")
args = parser.parse_args()


if __name__ == '__main__':
    # ── Read image list and labels from label_file ──
    images = []
    labels_dict = {}
    label_list = ["0", "90", "180", "270"]

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

    # ── Init VSX model ──
    text_reco = Classification(
        model_prefix_path=args.model_prefix_path,
        vdsp_params_info=args.vdsp_params_info,
        device_id=args.device_id,
        batch_size=args.batch,
        balance_mode=0,
        is_async_infer=False,
        model_output_op_name="",
    )

    all_preds, all_labels = [], []
    time_begin = time.time()

    with open(args.output_file, "w") as outfile:
        for img_path in tqdm.tqdm(images, desc="Evaluating"):
            if not os.path.exists(img_path):
                print(f"Warning: image not found, skipping: {img_path}")
                continue

            basename = os.path.splitext(os.path.basename(img_path))[0]
            gt_label = labels_dict.get(os.path.basename(img_path), -1)
            all_labels.append(gt_label)

            result = text_reco.run_sync(img_path)
            features = np.squeeze(result[0]["features"])
            if features.ndim == 1:
                features = features.reshape(1, -1)
            all_preds.append(features)

            pred_idxs = features.argmax(axis=1)
            decode_out = [(label_list[idx], float(features[0, idx])) for idx in pred_idxs]
            print(f"{basename}: VSX={decode_out}")
            outfile.write(f"{basename} VSX={decode_out}\n")

    time_end = time.time()
    text_reco.finish()

    # ── Compute Top-1 accuracy ──
    if len(all_labels) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.array(all_labels)
        acc = topk_accuracy(all_preds, all_labels, topk=(1,))
        print("\n========== Top-1 Accuracy ==========")
        print(f"VSX   Top-1: {acc[1]:.2f}%")
        print(f"Total : {len(all_labels)}")
        print(f"Time  : {time_end - time_begin:.2f}s ({len(all_labels) / (time_end - time_begin):.2f} img/s)")
        print("====================================\n")

'''
一些测试结论：
- paddle官方预处理为: target_short_edge=256, crop=224
- vacc没有target_short_edge操作, 尝试resize和resize-crop, 实测直接resze优于resize-crop
- fp16, vacc和onnx对齐
- int8, mse量化方式精度最好
'''


'''
unset VACM_LOG_CFG
FP16
deploy_weights/PP-LCNet_x1_0_doc_ori_infer_inference_sim_vacc

vacc_deploy_cls/vdsp_crop_resize.json
========== Top-1 Accuracy ==========
VSX   Top-1: 74.59%
Total : 2593
Time  : 9.98s (259.84 img/s)
====================================

vacc_deploy_cls/vdsp_resize.json
========== Top-1 Accuracy ==========
VSX   Top-1: 75.78%
Total : 2593
Time  : 9.93s (261.21 img/s)
====================================
'''



'''
deploy_weights/PP-LCNet_x1_0_doc_ori_infer_inference_sim_int8_vacc/mod

vacc_deploy_cls/vdsp_crop_resize.json
========== Top-1 Accuracy ==========
VSX   Top-1: 69.26%
Total : 2593
Time  : 9.47s (273.95 img/s)
===================================


vacc_deploy_cls/vdsp_resize.json
========== Top-1 Accuracy ==========
VSX   Top-1: 70.34%
Total : 2593
Time  : 9.15s (283.49 img/s)
====================================
'''

'''
vacc_deploy_cls/vdsp_resize.json
deploy_weights/PP-LCNet_x1_0_doc_ori_infer_inference_sim_vacc_int8_kl_divergence/mod
========== Top-1 Accuracy ==========
VSX   Top-1: 71.42%
Total : 2593
Time  : 9.47s (273.89 img/s)
====================================

deploy_weights/PP-LCNet_x1_0_doc_ori_infer_inference_sim_vacc_int8_max/mod
========== Top-1 Accuracy ==========
VSX   Top-1: 64.33%
Total : 2593
Time  : 9.63s (269.14 img/s)
====================================

deploy_weights/PP-LCNet_x1_0_doc_ori_infer_inference_sim_vacc_int8_mse/mod
========== Top-1 Accuracy ==========
VSX   Top-1: 71.65%
Total : 2593
Time  : 9.53s (272.09 img/s)
====================================
'''
