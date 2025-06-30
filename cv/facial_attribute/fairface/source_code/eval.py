import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import argparse


def plot_confusion_matrix(cm, classes, 
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in zip(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def preprocess_val_label(csv_path):
    df = pandas.read_csv(csv_path)

    df["age_id"] = df["age"].astype("category").cat.codes
    df["race_id"] = df["race"].astype("category").cat.codes
    df["gender_id"] = df["gender"].astype("category").cat.codes

    # age
    df.loc[df['age'] == '0-2', 'age_id'] = 0
    df.loc[df['age'] == '3-9', 'age_id'] = 1
    df.loc[df['age'] == '10-19', 'age_id'] = 2
    df.loc[df['age'] == '20-29', 'age_id'] = 3
    df.loc[df['age'] == '30-39', 'age_id'] = 4
    df.loc[df['age'] == '40-49', 'age_id'] = 5
    df.loc[df['age'] == '50-59', 'age_id'] = 6
    df.loc[df['age'] == '60-69', 'age_id'] = 7
    df.loc[df['age'] == 'more than 70', 'age_id'] = 8

    # race
    df.loc[df['race'] == 'White', 'race_id'] = 0
    df.loc[df['race'] == 'Black', 'race_id'] = 1
    df.loc[df['race'] == 'Latino_Hispanic', 'race_id'] = 2
    df.loc[df['race'] == 'East Asian', 'race_id'] = 3
    df.loc[df['race'] == 'Southeast Asian', 'race_id'] = 4
    df.loc[df['race'] == 'Indian', 'race_id'] = 5
    df.loc[df['race'] == 'Middle Eastern', 'race_id'] = 6

    # gender
    df.loc[df['gender'] == 'Male', 'gender_id'] = 0
    df.loc[df['gender'] == 'Female', 'gender_id'] = 1

    # new file
    new_path = csv_path.replace(".csv", "_onehot.csv")
    df.to_csv(new_path, index=False)
    return new_path

parser = argparse.ArgumentParser(description="RUN EVAL")
parser.add_argument("--val_label_path", type = str, default = "/path/to/fairface_label_val.csv", help = "img dir path")
parser.add_argument("--val_pred_path", type = str, default = "/path/to/vsx_predict.csv", help = "model info")

args = parser.parse_args()

if __name__ == '__main__':

    val_label_path = args.val_label_path
    val_pred_path = args.val_pred_path

    # label preprocess, convert to id
    val_label_path_new = preprocess_val_label(val_label_path)
    df_label = pandas.read_csv(val_label_path_new)
    df_pred = pandas.read_csv(val_pred_path)


    d_race = {
    0:'White',
    1:'Black',
    2:'Latino_Hispanic',
    3:'East Asian',
    4:'Southeast Asian',
    5:'Indian',
    6:'Middle Eastern'
    }



    d_gender = {
    0: 'Male',
    1: 'Female'
    }

    d_age = {
    0:'0-2',
    1:'3-9',
    2:'10-19',
    3:'20-29',
    4:'30-39',
    5:'40-49',
    6:'50-59',
    7:'60-69',
    8:'70+'
    }


    age  = list(d_age.values())
    race = list(d_race.values())
    gender = list(d_gender.values())

    for item in ['age_id', 'gender_id', 'race_id']:
        labels, predicts = df_label[item], df_pred[item]
        #cnf_matrix=confusion_matrix(labels, predicts)
        #plt.rcParams['figure.figsize'] = (16.0, 6.0)
        #plt.figure()
        #plot_confusion_matrix(cnf_matrix, classes=eval(item.replace("_id", "")),  normalize=True, title="normalized")
        #plt.savefig("{}.png".format(item))
        print(classification_report(labels, predicts, digits=4, target_names=eval(item.replace("_id", ""))))  # eval()，python内定函数，通过字符串寻找同名变量


