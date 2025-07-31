# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import cv2
import time
import json
import glob
import argparse
import threading
import numpy as np
import pandas as pd
from tqdm.contrib import tzip
from queue import Queue
from threading import Thread, Event
from typing import Dict, Generator, Iterable, List, Union

import vaststreamx as vsx


class VSXInference:
    def __init__(self,
        model_prefix_path: Union[str, Dict[str, str]],
        vdsp_params_info: str,
        device_id: int = 0,
        batch_size: int = 1,
        is_async_infer: bool = False,
        model_output_op_name: str = "", ) -> None:


        if isinstance(vdsp_params_info, str):
            with open(vdsp_params_info) as f:
                vdsp_params_info_dict = json.load(f)
        
        self.model_size = vdsp_params_info_dict["OpConfig"]["OimageWidth"]
        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.preprocess_name = "preprocess_res"
        self.input_id = 0

        self.attr = vsx.AttrKey
        assert vsx.set_device(self.device_id)==0
        # 构建model，模型三件套目录
        model_path = model_prefix_path
        self.model = vsx.Model(model_path, batch_size)
        # 输入预处理op
        # RGB_PLANAR
        self.fusion_op = vsx.Operator.load_ops_from_json_file(vdsp_params_info)[0]
        
        # 构建graph
        self.graph = vsx.Graph()
        self.model_op = vsx.ModelOperator(self.model)
        self.graph.add_operators(self.fusion_op, self.model_op)#

        # 构建stream
        self.infer_stream = vsx.Stream(self.graph, vsx.StreamBalanceMode.ONCE)
        if is_async_infer and len(self.model_output_op_name) > 0:
            self.infer_stream.register_operator_output(self.model_output_op_name, self.model_op)
        else:
            self.infer_stream.register_operator_output(self.model_op)
        
        # 预处理算子输出
        n,c,h,w = self.model.input_shape[0]
        self.infer_stream.register_operator_output(self.preprocess_name, self.fusion_op, [[(c,h,w), vsx.TypeFlag.FLOAT16]])

        self.infer_stream.build()
        
        self.current_id = -1
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}
        # 异步
        self.consumer = Thread(target=self.async_receive_infer)
        self.consumer.start()

    
    def async_receive_infer(self, ):
        while 1:
            try:
                result = None
                if len(self.model_output_op_name) > 0:
                    result = self.infer_stream.get_operator_output(self.model_output_op_name)
                else:
                    result = self.infer_stream.get_operator_output(self.model_op)
                if result is not None:
                    pre_process_tensor = self.infer_stream.get_operator_output(self.preprocess_name)
                    # 输出顺序和输入一致 
                    self.current_id += 1
                    input_id,height, width = self.input_dict[self.current_id]
                    model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in result[0]] ]
                    self.result_dict[input_id] = []
                    self.post_processing(input_id, height, width, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break
    
    def finish(self,):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()

    def post_processing(self, input_id, height, width, stream_output_list):
        output_data = stream_output_list[0][0]
        self.result_dict[input_id] = output_data

    def _run(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)

        input_id = self.input_id
        self.input_dict[input_id] = (input_id, height, width)
        self.event_dict[input_id] = Event()


        self.infer_stream.run_async([input_image])
        
        self.input_id += 1

        return input_id

    def run(self, image:Union[str, np.ndarray]):
        input_id = self._run(image)
        self.event_dict[input_id].wait()
        result = self.result_dict.pop(input_id)
        del self.event_dict[input_id]

        return result

    def run_batch(self, images: Iterable[Union[str, np.ndarray]]) -> Generator[str, None, None]:

        queue = Queue(20)
        
        def input_thread():
            for image in images:
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
    
    def run_sync(self, image:Union[str, np.ndarray]):
        if isinstance(image, str):
            cv_image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = np.stack(cv2.split(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))
        assert len(image.shape) == 3
        c, height, width = image.shape
        assert c == 3
        
        input_image = vsx.create_image(image, vsx.ImageFormat.RGB_PLANAR, width, height, self.device_id)

        output = self.infer_stream.run_sync([input_image])
        model_output_list = [ [vsx.as_numpy(out)[0].astype(np.float32) for out in output[0]] ]
        output_data = model_output_list[0][0]

        return output_data
        

def save_result_csv(files, results, save_csv_path):
    face_names = []
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []

    for file, output in tzip(files, results):
        face_names.append('val/' + os.path.basename(file))

        outputs = np.squeeze(output)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)


    age_id, race_id, gender_id = age_preds_fair, race_preds_fair, gender_preds_fair

    result = pd.DataFrame([face_names,
                        race_preds_fair,
                        gender_preds_fair,
                        age_preds_fair,
                        age_id,race_id,gender_id,
                        race_scores_fair,
                        gender_scores_fair,
                        age_scores_fair, ]).T
    result.columns = ['file',
                    'race_preds_fair',
                    'gender_preds_fair',
                    'age_preds_fair',
                    'age_id', 'race_id', 'gender_id',
                    'race_scores_fair',
                    'gender_scores_fair',
                    'age_scores_fair']
    result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
    result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
    result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
    result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
    result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
    result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian'
    result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'


    # gender
    result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
    result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

    # age
    result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
    result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
    result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
    result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
    result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
    result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49'
    result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
    result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
    result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'

    result[['file',
            'race',
            'gender', 'age',
            'age_id', 'race_id', 'gender_id',
            'race_scores_fair', 
            'gender_scores_fair', 'age_scores_fair']].to_csv(save_csv_path)

    print("saved results at ", save_csv_path)


def set_config():
    parse = argparse.ArgumentParser(description="RUN WITH VSX")
    parse.add_argument("--file_path",type=str,default="/path/to/fairface/val",help="img dir",)
    parse.add_argument("--model_prefix_path",type=str,default="deploy_weights/official_fairface_run_stream_fp16/mod",help="model info")
    parse.add_argument("--vdsp_params_info",type=str,default="../vacc_code/vdsp_params/official-fairface_res34-vdsp_params.json",help="vdsp op info",)
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument("--batch", type=int, default=1, help="bacth size")
    parse.add_argument("--save_file", type=str, default="vsx_predict.csv", help="save result")
    args = parse.parse_args()

    return args


if __name__ == '__main__':
    args = set_config()

    vsx_inference = VSXInference(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )

    # Test multiple images
    image_files = glob.glob(os.path.join(args.file_path, "*.jpg"))
    # sort images
    image_files.sort(key=lambda x: int(os.path.basename(x)[:-4]))

    results = vsx_inference.run_batch(image_files)
    
    predice_results = []
    for (image_path, result) in tzip(image_files, results):
        # print(f"{image_path}")
        predice_results.append(result)
    save_result_csv(image_files, predice_results, args.save_file)

    vsx_inference.finish()
    
"""
fairface_res34-int8-percentile-1_3_224_224-vacc

              precision    recall  f1-score   support

         0-2     0.7438    0.7588    0.7512       199
         3-9     0.8018    0.8326    0.8169      1356
       10-19     0.5749    0.4615    0.5120      1181
       20-29     0.6552    0.7336    0.6922      3300
       30-39     0.5077    0.4966    0.5021      2330
       40-49     0.4860    0.4486    0.4666      1353
       50-59     0.5155    0.5000    0.5077       796
       60-69     0.4698    0.4611    0.4654       321
         70+     0.5647    0.4068    0.4729       118

    accuracy                         0.6029     10954
   macro avg     0.5911    0.5666    0.5763     10954
weighted avg     0.5975    0.6029    0.5986     10954

              precision    recall  f1-score   support

        Male     0.9489    0.9451    0.9470      5792
      Female     0.9387    0.9429    0.9408      5162

    accuracy                         0.9440     10954
   macro avg     0.9438    0.9440    0.9439     10954
weighted avg     0.9441    0.9440    0.9440     10954

                 precision    recall  f1-score   support

          White     0.7658    0.7794    0.7725      2085
          Black     0.8822    0.8612    0.8715      1556
Latino_Hispanic     0.5593    0.5786    0.5687      1623
     East Asian     0.7376    0.7871    0.7615      1550
Southeast Asian     0.6594    0.6403    0.6497      1415
         Indian     0.7715    0.7263    0.7482      1516
 Middle Eastern     0.6548    0.6385    0.6466      1209

       accuracy                         0.7215     10954
      macro avg     0.7186    0.7159    0.7170     10954
   weighted avg     0.7225    0.7215    0.7217     10954


"""

"""
face_attribute-fp16-none-1_3_224_224-vacc

              precision    recall  f1-score   support

         0-2     0.7550    0.7588    0.7569       199
         3-9     0.8063    0.8319    0.8189      1356
       10-19     0.5812    0.4759    0.5233      1181
       20-29     0.6567    0.7309    0.6918      3300
       30-39     0.5052    0.5013    0.5032      2330
       40-49     0.4924    0.4531    0.4719      1353
       50-59     0.5170    0.4962    0.5064       796
       60-69     0.4760    0.4642    0.4700       321
         70+     0.5926    0.4068    0.4824       118

    accuracy                         0.6049     10954
   macro avg     0.5980    0.5688    0.5805     10954
weighted avg     0.6002    0.6049    0.6011     10954

              precision    recall  f1-score   support

        Male     0.9496    0.9439    0.9467      5792
      Female     0.9375    0.9438    0.9406      5162

    accuracy                         0.9439     10954
   macro avg     0.9435    0.9439    0.9437     10954
weighted avg     0.9439    0.9439    0.9439     10954

                 precision    recall  f1-score   support

          White     0.7667    0.7722    0.7694      2085
          Black     0.8783    0.8625    0.8703      1556
Latino_Hispanic     0.5630    0.5810    0.5719      1623
     East Asian     0.7373    0.7839    0.7598      1550
Southeast Asian     0.6549    0.6424    0.6486      1415
         Indian     0.7718    0.7230    0.7466      1516
 Middle Eastern     0.6460    0.6385    0.6423      1209

       accuracy                         0.7200     10954
      macro avg     0.7168    0.7148    0.7156     10954
   weighted avg     0.7211    0.7200    0.7203     10954
   
"""