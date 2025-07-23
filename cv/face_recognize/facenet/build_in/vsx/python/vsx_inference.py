import os
import cv2
import time
import json
import glob
import math
import argparse
import numpy as np
import os.path as osp
from tqdm.contrib import tzip
from queue import Queue
from threading import Thread, Event
from typing import Dict, Generator, Iterable, List, Union

from sklearn import metrics
from scipy.optimize import brentq

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
        output_data = stream_output_list[0][-1]
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

def set_config():
    parse = argparse.ArgumentParser(description="RUN WITH VSX")
    parse.add_argument(
        "--image_dir",
        type=str,
        default="./dataset/face/lfw_mtcnnpy_160",
        help="image_dir img dir",
    )
    parse.add_argument(
        "--lfw_pairs",
        type=str,
        default="./dataset/face/pairs.txt",
        help="pairs.txt file path",
    )
    parse.add_argument(
        "--model_prefix_path",
        type=str,
        default="deploy_weights/facenet_casia_webface-int8-percentile-1_3_160_160-vacc/mod",
        help="model info")
    parse.add_argument(
        "--vdsp_params_info",
        type=str,
        default="face_recognize/facenet/vacc_code/vdsp_params/pytorch-facenet_vggface2-vdsp_params.json",
        help="vdsp op info",
    )
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument("--batch", type=int, default=1, help="bacth size")
    parse.add_argument("--save_dir", type=str, default="./vsx_results", help="save result")
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
    
    os.makedirs(args.save_dir, exist_ok=True)

    # Test multiple images
    image_files = glob.glob(f"{args.image_dir}/*/*", recursive=True)
    # sort images
    image_files.sort()

    pairs = read_pairs(args.lfw_pairs)
    path_list, issame_list = get_paths(args.image_dir, pairs)
    l2_norm = False
    embeddings_dict = {}

    # run vsx
    results = vsx_inference.run_batch(image_files)
    for (image_path, result) in tzip(image_files, results):
        # post process
        embeddings_dict[image_path] = result

    embeddings = np.array([embeddings_dict[path] for path in path_list])
    tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings, issame_list)
    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1.0 - x - interpolate.interp1d(fpr, tpr)(x), 0.0, 1.0)

    print("Accuracy: %2.5f+-%2.5f" % (np.mean(accuracy), np.std(accuracy)))
    print("Validation rate: %2.5f+-%2.5f @ FAR=%2.5f" % (val, val_std, far))
    print("Area Under Curve (AUC): %1.5f" % auc)
    print("Equal Error Rate (EER): %1.5f" % eer)
    vsx_inference.finish()
    
"""
pytorch

facenet_vggface2-int8-percentile-1_3_160_160-vacc
Accuracy: 0.99350+-0.00411
Validation rate: 0.97633+-0.01269 @ FAR=0.00067
Area Under Curve (AUC): 0.99959
Equal Error Rate (EER): 0.00567

facenet_casia_webface-int8-percentile-1_3_160_160-vacc
Accuracy: 0.98867+-0.00581
Validation rate: 0.94900+-0.01606 @ FAR=0.00100
Area Under Curve (AUC): 0.99869
Equal Error Rate (EER): 0.01267

"""