# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import math
import glob
import argparse
import os.path as osp
import numpy as np
from tqdm.contrib import tzip

from sklearn import metrics
from scipy.optimize import brentq

parse = argparse.ArgumentParser(description="IMAGENET TOPK")
parse.add_argument("--gt_dir", type=str, default="./dataset/face/lfw_mtcnnpy_160", help="src image folder")
parse.add_argument("--gt_pairs_path", type=str, default="./dataset/face/pairs.txt", help="https://github.com/davidsandberg/facenet/blob/master/data/pairs.txt")
parse.add_argument("--input_npz_path", type=str, default="npz_datalist.txt", help="npz text file")
parse.add_argument("--out_npz_dir", type=str, default="./code/vamc/vamp/0.2.0/vamp_result", help="vamp output folder")

args = parse.parse_args()
print(args)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, "r") as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs, dtype=object)

def add_extension(path):
    if osp.exists(path + ".jpg"):
        return path + ".jpg"
    elif osp.exists(path + ".png"):
        return path + ".png"
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(
                osp.join(lfw_dir, pair[0], pair[0] + "_" + "%04d" % int(pair[1]))
            )
            path1 = add_extension(
                osp.join(lfw_dir, pair[0], pair[0] + "_" + "%04d" % int(pair[2]))
            )
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(
                osp.join(lfw_dir, pair[0], pair[0] + "_" + "%04d" % int(pair[1]))
            )
            path1 = add_extension(
                osp.join(lfw_dir, pair[2], pair[2] + "_" + "%04d" % int(pair[3]))
            )
            issame = False
        if osp.exists(path0) and osp.exists(
            path1
        ):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print("Skipped %d image pairs" % nrof_skipped_pairs)

    return path_list, issame_list

from sklearn.model_selection import KFold
from scipy import interpolate

# LFW functions taken from David Sandberg's FaceNet implementation
def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    is_false_positive = []
    is_false_negative = []

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _ ,_ = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
        is_false_positive.extend(is_fp)
        is_false_negative.extend(is_fn)

    return tpr, fpr, accuracy, is_false_positive, is_false_negative

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc, is_fp, is_fn

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, fp, fn  = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far, fp, fn

def decode():
    lfw_aligned_dir = args.gt_dir
    lfw_pairs = args.gt_pairs_path
    pairs = read_pairs(lfw_pairs)
    path_list, issame_list = get_paths(lfw_aligned_dir, pairs)
    embeddings_dict = {}

    with open(args.input_npz_path) as f:
        input_npz_list = [cls.strip() for cls in f.readlines()]
    npz_result = glob.glob(args.out_npz_dir + '/*')
    npz_result.sort()

    for file, result in tzip(input_npz_list, npz_result):
        pred = np.load(result, allow_pickle=True)["output_0"]
        key = lfw_aligned_dir + '/' + '/'.join(file.split('/')[-2:])
        embeddings_dict[key.replace('.npz', '.png')] = pred[0]

    embeddings = np.array([embeddings_dict[path] for path in path_list])
    tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings, issame_list)
    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1.0 - x - interpolate.interp1d(fpr, tpr)(x), 0.0, 1.0)

    print("Accuracy: %2.5f+-%2.5f" % (np.mean(accuracy), np.std(accuracy)))
    print("Validation rate: %2.5f+-%2.5f @ FAR=%2.5f" % (val, val_std, far))
    print("Area Under Curve (AUC): %1.5f" % auc)
    print("Equal Error Rate (EER): %1.5f" % eer)


if __name__ == "__main__":
    decode()

"""
facenet_vggface2-int8-kl_divergence-1-3-vacc
0.9944999999999998
"""