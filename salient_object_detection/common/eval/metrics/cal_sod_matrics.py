# -*- coding: utf-8 -*-

import os
from collections import defaultdict
from functools import partial
from multiprocessing import pool
from threading import RLock as TRLock

import numpy as np
from tqdm import tqdm

from utils.misc import get_gt_pre_with_name, get_name_list, make_dir
from utils.print_formatter import formatter_for_tabulate
from utils.recorders import (
    BINARY_METRIC_MAPPING,
    GRAYSCALE_METRICS,
    BinaryMetricRecorder,
    GrayscaleMetricRecorder,
    MetricExcelRecorder,
    TxtRecorder,
)


class Recorder:
    def __init__(
        self,
        method_names,
        dataset_names,
        metric_names,
        *,
        txt_path,
        to_append,
        xlsx_path,
        sheet_name,
    ):
        self.curves = defaultdict(dict)  # Two curve metrics
        self.metrics = defaultdict(dict)  # Six numerical metrics
        self.method_names = method_names
        self.dataset_names = dataset_names

        self.txt_recorder = None
        if txt_path:
            self.txt_recorder = TxtRecorder(
                txt_path=txt_path,
                to_append=to_append,
                max_method_name_width=max([len(x) for x in method_names]),  # 显示完整名字
            )

        self.excel_recorder = None
        if xlsx_path:
            excel_metric_names = []
            for x in metric_names:
                if x in GRAYSCALE_METRICS:
                    if x == "em":
                        excel_metric_names.append(f"max{x}")
                        excel_metric_names.append(f"avg{x}")
                        excel_metric_names.append(f"adp{x}")
                    else:
                        config = BINARY_METRIC_MAPPING[x]
                        if config["kwargs"]["with_dynamic"]:
                            excel_metric_names.append(f"max{x}")
                            excel_metric_names.append(f"avg{x}")
                        if config["kwargs"]["with_adaptive"]:
                            excel_metric_names.append(f"adp{x}")
                else:
                    excel_metric_names.append(x)

            self.excel_recorder = MetricExcelRecorder(
                xlsx_path=xlsx_path,
                sheet_name=sheet_name,
                row_header=["methods"],
                dataset_names=dataset_names,
                metric_names=excel_metric_names,
            )

    def record(self, method_results, dataset_name, method_name):
        """Record results"""
        method_curves = method_results.get("sequential")
        method_metrics = method_results["numerical"]
        self.curves[dataset_name][method_name] = method_curves
        self.metrics[dataset_name][method_name] = method_metrics

    def export(self):
        """After evaluating all methods, export results to ensure the order of names."""
        for dataset_name in self.dataset_names:
            if dataset_name not in self.metrics:
                continue

            for method_name in self.method_names:
                dataset_results = self.metrics[dataset_name]
                method_results = dataset_results.get(method_name)
                if method_results is None:
                    continue

                if self.txt_recorder:
                    self.txt_recorder.add_row(row_name="Dataset", row_data=dataset_name)
                    self.txt_recorder(method_results=method_results, method_name=method_name)
                if self.excel_recorder:
                    self.excel_recorder(
                        row_data=method_results, dataset_name=dataset_name, method_name=method_name
                    )


def cal_image_matrics(
    sheet_name: str = "results",
    txt_path: str = "",
    to_append: bool = True,
    xlsx_path: str = "",
    methods_info: dict = None,
    datasets_info: dict = None,
    curves_npy_path: str = "./curves.npy",
    metrics_npy_path: str = "./metrics.npy",
    num_bits: int = 3,
    num_workers: int = 2,
    ncols_tqdm: int = 79,
    metric_names: tuple = ("sm", "wfm", "mae", "fmeasure", "em"),
):
    """Save the results of all models on different datasets in a `npy` file in the form of a
    dictionary.

    Args:
        sheet_name (str, optional): The type of the sheet in xlsx file. Defaults to "results".
        txt_path (str, optional): The path of the txt for saving results. Defaults to "".
        to_append (bool, optional): Whether to append results to the original record. Defaults to True.
        xlsx_path (str, optional): The path of the xlsx file for saving results. Defaults to "".
        methods_info (dict, optional): The method information. Defaults to None.
        datasets_info (dict, optional): The dataset information. Defaults to None.
        curves_npy_path (str, optional): The npy file path for saving curve data. Defaults to "./curves.npy".
        metrics_npy_path (str, optional): The npy file path for saving metric values. Defaults to "./metrics.npy".
        num_bits (int, optional): The number of bits used to format results. Defaults to 3.
        num_workers (int, optional): The number of workers of multiprocessing or multithreading. Defaults to 2.
        ncols_tqdm (int, optional): Number of columns for tqdm. Defaults to 79.
        metric_names (tuple, optional): Names of metrics. Defaults to ("sm", "wfm", "mae", "fmeasure", "em").

    Returns:
        {
          dataset1:{
            method1:[fm, em, p, r],
            method2:[fm, em, p, r],
            .....
          },
          dataset2:{
            method1:[fm, em, p, r],
            method2:[fm, em, p, r],
            .....
          },
          ....
        }

    """
    if all([x in BinaryMetricRecorder.suppoted_metrics for x in metric_names]):
        metric_class = BinaryMetricRecorder
    elif all([x in GrayscaleMetricRecorder.suppoted_metrics for x in metric_names]):
        metric_class = GrayscaleMetricRecorder
    else:
        raise ValueError(metric_names)

    method_names = tuple(methods_info.keys())
    dataset_names = tuple(datasets_info.keys())
    recorder = Recorder(
        method_names=method_names,
        dataset_names=dataset_names,
        metric_names=metric_names,
        txt_path=txt_path,
        to_append=to_append,
        xlsx_path=xlsx_path,
        sheet_name=sheet_name,
    )

    # multi-process mode
    # tqdm.set_lock(RLock())
    # pool_cls = pool.Pool
    # multi-threading mode
    tqdm.set_lock(TRLock())
    pool_cls = pool.ThreadPool
    procs = pool_cls(processes=num_workers, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    print(f"Create a {procs}).")

    procs_idx = 0
    for dataset_name, dataset_path in datasets_info.items():
        # 获取真值图片信息
        gt_info = dataset_path["mask"]
        gt_root = gt_info["path"]
        gt_prefix = gt_info.get("prefix", "")
        gt_suffix = gt_info["suffix"]
        # 真值名字列表
        gt_index_file = dataset_path.get("index_file")
        if gt_index_file:
            gt_name_list = get_name_list(
                data_path=gt_index_file,
                name_prefix=gt_prefix,
                name_suffix=gt_suffix,
            )
        else:
            gt_name_list = get_name_list(
                data_path=gt_root,
                name_prefix=gt_prefix,
                name_suffix=gt_suffix,
            )
        assert len(gt_name_list) > 0, "there is not ground truth."

        # ==>> test the intersection between pre and gt for each method <<==
        for method_name, method_info in methods_info.items():
            method_root = method_info["path_dict"]
            method_dataset_info = method_root.get(dataset_name, None)
            if method_dataset_info is None:
                tqdm.write(f"{method_name} does not have results on {dataset_name}")
                continue

            # 预测结果存放路径下的图片文件名字列表和扩展名称
            pre_prefix = method_dataset_info.get("prefix", "")
            pre_suffix = method_dataset_info["suffix"]
            pre_root = method_dataset_info["path"]
            pre_name_list = get_name_list(
                data_path=pre_root, name_prefix=pre_prefix, name_suffix=pre_suffix
            )

            # get the intersection
            eval_name_list = sorted(list(set(gt_name_list).intersection(pre_name_list)))
            if len(eval_name_list) == 0:
                tqdm.write(f"{method_name} does not have results on {dataset_name}")
                continue

            kwargs = dict(
                names=eval_name_list,
                num_bits=num_bits,
                pre_root=pre_root,
                pre_prefix=pre_prefix,
                pre_suffix=pre_suffix,
                gt_root=gt_root,
                gt_prefix=gt_prefix,
                gt_suffix=gt_suffix,
                desc=f"[{dataset_name}({len(gt_name_list)}):{method_name}({len(pre_name_list)})]",
                proc_idx=procs_idx,
                metric_names=metric_names,
                ncols_tqdm=ncols_tqdm,
                metric_class=metric_class,
            )
            callback = partial(recorder.record, dataset_name=dataset_name, method_name=method_name)
            procs.apply_async(func=evaluate, kwds=kwargs, callback=callback)
            # for debugging
            # callback(evaluate(**kwargs), dataset_name=dataset_name, method_name=method_name)
            procs_idx += 1
    procs.close()
    procs.join()

    recorder.export()
    if curves_npy_path:
        make_dir(os.path.dirname(curves_npy_path))
        np.save(curves_npy_path, recorder.curves)
        tqdm.write(f"All curves has been saved in {curves_npy_path}")
    if metrics_npy_path:
        make_dir(os.path.dirname(metrics_npy_path))
        np.save(metrics_npy_path, recorder.metrics)
        tqdm.write(f"All metrics has been saved in {metrics_npy_path}")
    formatted_string = formatter_for_tabulate(recorder.metrics, method_names, dataset_names)
    tqdm.write(f"All methods have been evaluated:\n{formatted_string}")


def evaluate(
    names,
    num_bits,
    gt_root,
    gt_prefix,
    gt_suffix,
    pre_root,
    pre_prefix,
    pre_suffix,
    metric_class,
    desc="",
    proc_idx=None,
    metric_names=None,
    ncols_tqdm=79,
):
    metric_recoder = metric_class(metric_names=metric_names)
    # https://github.com/tqdm/tqdm#parameters
    # https://github.com/tqdm/tqdm/blob/master/examples/parallel_bars.py
    for name in tqdm(
        names, total=len(names), desc=desc, position=proc_idx, ncols=ncols_tqdm, lock_args=(False,)
    ):
        gt, pre = get_gt_pre_with_name(
            img_name=name,
            pre_root=pre_root,
            pre_prefix=pre_prefix,
            pre_suffix=pre_suffix,
            gt_root=gt_root,
            gt_prefix=gt_prefix,
            gt_suffix=gt_suffix,
            to_normalize=False,
        )
        metric_recoder.step(pre=pre, gt=gt, gt_path=os.path.join(gt_root, name))

    method_results = metric_recoder.show(num_bits=num_bits, return_ndarray=False)
    return method_results
