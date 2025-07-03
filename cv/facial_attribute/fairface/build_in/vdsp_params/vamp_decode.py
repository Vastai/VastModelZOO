import os
import cv2
import glob
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from tqdm.contrib import tzip


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


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="VAMP EVAL")
    parse.add_argument("--gt_dir", type=str, default="./dataset/face/fairface/val", help="hr gt image folder")
    parse.add_argument("--input_npz_path", type=str, default="./dataset/face/fairface/npz_datalist.txt", help="npz text file")
    parse.add_argument("--out_npz_dir", type=str, default="./code/vamc/vamp/2.1.0/outputs", help="vamp output folder")
    parse.add_argument("--input_shape", nargs='+', type=int, default=[224, 224], help="vamp input shape")
    parse.add_argument("--draw_dir", type=str, default="./npz_draw_result", help="vamp npz draw image reult folder")
    parse.add_argument("--vamp_flag", action='store_true', default=True, help="decode vamp or vamc result")

    args = parse.parse_args()
    print(args)

    os.makedirs(args.draw_dir, exist_ok=True)

    output_npz_list = glob.glob(os.path.join(args.out_npz_dir, "*.npz"))
    output_npz_list.sort()
    
    with open(args.input_npz_path, 'r') as f:
        file_lines = f.readlines()
        # file_lines.sort(key=lambda x: int(os.path.basename(x.strip('\n'))[:-4]))
        
        image_list = []
        predict_list = []
        for i, line in enumerate(tqdm(file_lines)):
            npz_name = os.path.basename(line.strip('\n'))
            file_name = npz_name.replace(".npz", ".jpg")

            # load npy
            if args.vamp_flag:
                npz_file = output_npz_list[i]
            else:
                npz_file = os.path.join(args.out_npz_dir, npz_name)

            heatmap = np.load(npz_file, allow_pickle=True)["output_0"].astype(np.float32)
            
            # draw
            output = np.squeeze(heatmap)
            predict_list.append(output)
            image_list.append(file_name)

    save_result_csv(image_list, predict_list, os.path.join(args.draw_dir, "npz_predict.csv"))

"""

face_attribute-int8-percentile-1_3_224_224-vacc

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
