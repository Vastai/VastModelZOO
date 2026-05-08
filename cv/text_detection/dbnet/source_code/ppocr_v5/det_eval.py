import cv2
import copy
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from eval_precision import DetectorConfig, Detector, TextDetMetric


def draw_boxes(img: np.ndarray, dt_boxes: list, gt_boxes: list, save_path: str):
    """gt-green, detect-red"""
    draw_img = copy.deepcopy(img)

    for box in dt_boxes:
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(draw_img, [pts], True, (0, 0, 255), 2)
    for box in gt_boxes:
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(draw_img, [pts], True, (0, 255, 0), 2)
    cv2.imwrite(save_path, draw_img)


if __name__ == "__main__":
    detector = Detector(
        config=DetectorConfig(
            path="PP-OCRv5_mobile_det_infer_inference_sim.onnx",
            use_fixed_shape=True, # set in False, correspond with official max_side_len=960; set in True, just set in [960, 960]

            vacc_path="deploy_weights/PP-OCRv5_mobile_det_infer_fp16_960/mod",
            vdsp_path="cv/text_detection/dbnet/build_in/vdsp_params/ppocr-ch_PP_OCRv5_det-vdsp_params.json" # just used in runstream
        ),
    )

    # dataset
    dataset = load_dataset("SWHL/text_det_test_dataset") # auto download from hf
    test_data = dataset["test"]
    # dataset = load_dataset("json", data_files="text_det_test_dataset/data/test/metadata.jsonl")
    # test_data = dataset["train"]

    content = []
    content_vacc = []
    pred_path = "det_pred_onnx.txt"
    pred_vacc_path = "det_pred_vacc.txt"

    # infer image
    for i, one_data in enumerate(tqdm(test_data, desc="infer images")):
        img = np.array(one_data.get("image"))

        detector.load_onnx_model()
        detector.load_vacc_model()     # runstream

        dt_boxes, scores = detector.detect_onnx(img)
        dt_boxes_vacc, scores_vacc = detector.detect_vacc(img)    # runstream

        dt_boxes = [] if dt_boxes is None else dt_boxes[0].tolist()
        dt_boxes_vacc = [] if dt_boxes_vacc is None else dt_boxes_vacc[0].tolist()

        elapse = 0
        gt_boxes = [v["points"] for v in one_data["shapes"]]
        content.append(f"{dt_boxes}\t{gt_boxes}\t{elapse}")
        content_vacc.append(f"{dt_boxes_vacc}\t{gt_boxes}\t{elapse}")

        # draw boxes
        if i < 10:
            img_draw1 = copy.deepcopy(img)
            img_draw2 = copy.deepcopy(img)
            draw_boxes(img_draw1, dt_boxes, gt_boxes, f"output_{i}_onnx.jpg")
            draw_boxes(img_draw2, dt_boxes_vacc, gt_boxes, f"output_{i}_vacc.jpg")

    with open(pred_path, "w", encoding="utf-8") as f:
        for v in content:
            f.write(f"{v}\n")

    with open(pred_vacc_path, "w", encoding="utf-8") as f:
        for v in content_vacc:
            f.write(f"{v}\n")

    # eval metric
    metric = TextDetMetric()
    metric_res = metric(pred_path)
    print('onnx metric: ', metric_res)

    metric_res = metric(pred_vacc_path)
    print('vacc metric: ', metric_res)



'''
# for dataset
export HF_ENDPOINT=https://hf-mirror.com
python cv/text_detection/dbnet/source_code/ppocr_v5/det_eval.py

# not show log
unset VACM_LOG_CFG

https://github.com/SWHL/TextDetMetric/tree/main
https://huggingface.co/datasets/SWHL/text_det_test_dataset
212 images

pred.txt like this file: https://github.com/SWHL/TextDetMetric/blob/main/pred.txt
'''


'''
PP-OCRv5_mobile

onnx/PP-OCRv5_mobile_det_infer_inference_sim.onnx
use_fixed_shape=False
{'precision': 0.7915, 'recall': 0.8266, 'hmean': 0.8087, 'avg_elapse': 0.0}

onnx/PP-OCRv5_mobile_det_infer_inference_sim.onnx
use_fixed_shape=True
{'precision': 0.7504, 'recall': 0.7922, 'hmean': 0.7707, 'avg_elapse': 0.0}

deploy_weights0506/PP-OCRv5_mobile_det_infer_inference_sim_fp16_runstream_960
use_fixed_shape=True
{'precision': 0.7545, 'recall': 0.7995, 'hmean': 0.7763, 'avg_elapse': 0.0}
'''

'''
PP-OCRv5_server_det

onnx/PP-OCRv5_server_det_infer_inference_sim.onnx
use_fixed_shape=False
{'precision': 0.8293, 'recall': 0.8667, 'hmean': 0.8476, 'avg_elapse': 0.0}

onnx/PP-OCRv5_server_det_infer_inference_sim.onnx
use_fixed_shape=True
{'precision': 0.8105, 'recall': 0.8346, 'hmean': 0.8224, 'avg_elapse': 0.0}

deploy_weights0506/PP-OCRv5_server_det_infer_inference_sim_fp16_runstream_960
use_fixed_shape=True
{'precision': 0.8086, 'recall': 0.8231, 'hmean': 0.8158, 'avg_elapse': 0.0}
'''
