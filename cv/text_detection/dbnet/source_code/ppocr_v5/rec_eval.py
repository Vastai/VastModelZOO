import cv2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from eval_precision import RecognizerConfig, Recognizer, TextRecMetric


if __name__ == "__main__":
    recognizer = Recognizer(
        config=RecognizerConfig(
            path="PP-OCRv5_mobile_rec_infer_inference_sim.onnx",
            dict_path="ppocrv5_onnx/data/dict/ppocrv5_dict.txt",
            use_fixed_shape=True, # set in False, correspond with official h_side_len=48, w_side_len dynamic; set in True, just set in [48, 320]
            
            vacc_path="deploy_weights/PP-OCRv5_mobile_rec_infer_fp16_320/mod",
            vdsp_path="cv/text_recognition/ppocr_v5_rec/build_in/vdsp_params/ppocr-ch_PP_OCRv5_rec-vdsp_params.json" # just used in runstream
        ),
    )

    # dataset
    dataset = load_dataset("SWHL/text_rec_test_dataset") # auto download from hf
    test_data = dataset["test"]
    # dataset = load_dataset("json", data_files="text_rec_test_dataset/data/test/metadata.jsonl")
    # test_data = dataset["train"]

    content = []
    content_vacc = []

    pred_path = "rec_pred_onnx.txt"
    pred_path_vacc = "rec_pred_vacc.txt"

    # infer image
    for i, one_data in enumerate(tqdm(test_data, desc="infer images")):
        img = np.array(one_data.get("image"))

        recognizer.load_onnx_model()
        recognizer.load_vacc_model()     # runstream

        rec_res = recognizer.recognize_onnx(img) # [('text', scores)]
        rec_res_vacc = recognizer.recognize_vacc(img) # [('text', scores)]

        rec_res = "" if len(rec_res)==0 else rec_res[0][0]
        rec_res_vacc = "" if len(rec_res_vacc)==0 else rec_res_vacc[0][0]

        elapse = 0

        gt = one_data.get("label", None)
        content.append(f"{rec_res}\t{gt}\t{elapse}")
        content_vacc.append(f"{rec_res_vacc}\t{gt}\t{elapse}")

        if i < 10:
            print(f'onnx: {rec_res}', f'vacc: {rec_res_vacc}')

    with open(pred_path, "w", encoding="utf-8") as f:
        for v in content:
            f.write(f"{v}\n")

    with open(pred_path_vacc, "w", encoding="utf-8") as f:
        for v in content_vacc:
            f.write(f"{v}\n")

    # eval metric
    metric = TextRecMetric()
    metric_res = metric(pred_path)
    print('onnx metric: ', metric_res)

    metric_res = metric(pred_path_vacc)
    print('vacc metric: ', metric_res)

'''
https://github.com/SWHL/TextRecMetric
https://huggingface.co/datasets/SWHL/text_rec_test_dataset

310 images

export HF_ENDPOINT=https://hf-mirror.com
pip install Levenshtein -i https://pypi.mirrors.ustc.edu.cn/simple/

pred.txt like this file: https://github.com/SWHL/TextRecMetric/blob/main/pred.txt
'''


'''
onnx/PP-OCRv5_mobile_rec_infer_inference_sim.onnx
use_fixed_shape=False
{'ExactMatch': 0.729, 'CharMatch': 0.9123, 'avg_elapse': 0.0}

onnx/PP-OCRv5_mobile_rec_infer_inference_sim.onnx
use_fixed_shape=True
{'ExactMatch': 0.6903, 'CharMatch': 0.8546, 'avg_elapse': 0.0}


deploy_weights0506/PP-OCRv5_mobile_rec_infer_inference_sim_fp16_runtream_320
use_fixed_shape=True
{'ExactMatch': 0.7258, 'CharMatch': 0.8687, 'avg_elapse': 0.0}
'''

'''
onnx/PP-OCRv5_server_rec_infer_inference_sim.onnx
use_fixed_shape=False
{'ExactMatch': 0.8032, 'CharMatch': 0.9376, 'avg_elapse': 0.0}

onnx/PP-OCRv5_server_rec_infer_inference_sim.onnx
use_fixed_shape=True
{'ExactMatch': 0.7548, 'CharMatch': 0.8759, 'avg_elapse': 0.0}

deploy_weights0506/PP-OCRv5_server_rec_infer_inference_sim_fp16_runtream_320
use_fixed_shape=True
{'ExactMatch': 0.7871, 'CharMatch': 0.8979, 'avg_elapse': 0.0}
'''
