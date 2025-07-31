# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import dlib
import torch
import torchvision
import pandas as pd
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms


def face_model(checkpoint):
    model = torchvision.models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device)
    model.eval()

    return model


def predidct_age_gender_race(imgs_path, save_prediction_at):
    img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path)]
    img_names.sort(key=lambda x: int(os.path.basename(x)[:-4]))
    # img_names = img_names[:100]

    # img path of face images
    face_names = []
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []

    for index, img_name in enumerate(img_names):
        if index % 1000 == 0:
            print("Predicting... {}/{}".format(index, len(img_names)))

        face_names.append('val/' + os.path.basename(img_name))
        image = dlib.load_rgb_image(img_name)
        image = transformer(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(device)

        # fair
        outputs = model(image)

        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

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
            'gender_scores_fair', 'age_scores_fair']].to_csv(save_prediction_at)

    print("saved results at ", save_prediction_at)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    face_dir = "./dataset/face/fairface/val" # aligened face
    save_result_path = "results.csv"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = face_model("res34_fair_align_multi_7_20190809.pt")

    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    predidct_age_gender_race(face_dir, save_result_path)

"""
# res34_fair_align_multi_7_20190809.pt

Normalized confusion matrix
              precision    recall  f1-score   support

         0-2     0.7550    0.7588    0.7569       199
         3-9     0.8057    0.8319    0.8186      1356
       10-19     0.5809    0.4742    0.5221      1181
       20-29     0.6556    0.7321    0.6918      3300
       30-39     0.5072    0.5013    0.5042      2330
       40-49     0.4924    0.4553    0.4731      1353
       50-59     0.5165    0.4912    0.5035       796
       60-69     0.4777    0.4673    0.4724       321
         70+     0.6125    0.4153    0.4949       118

    accuracy                         0.6052     10954
   macro avg     0.6004    0.5697    0.5820     10954
weighted avg     0.6004    0.6052    0.6012     10954

Normalized confusion matrix
              precision    recall  f1-score   support

        Male     0.9496    0.9434    0.9465      5792
      Female     0.9369    0.9438    0.9404      5162

    accuracy                         0.9436     10954
   macro avg     0.9433    0.9436    0.9434     10954
weighted avg     0.9436    0.9436    0.9436     10954

Normalized confusion matrix
                 precision    recall  f1-score   support

          White     0.7664    0.7775    0.7719      2085
          Black     0.8789    0.8631    0.8709      1556
Latino_Hispanic     0.5589    0.5816    0.5700      1623
     East Asian     0.7364    0.7839    0.7594      1550
Southeast Asian     0.6551    0.6417    0.6483      1415
         Indian     0.7714    0.7236    0.7468      1516
 Middle Eastern     0.6555    0.6311    0.6431      1209

       accuracy                         0.7204     10954
      macro avg     0.7175    0.7146    0.7158     10954
   weighted avg     0.7215    0.7204    0.7206     10954

"""