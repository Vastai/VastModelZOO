# -*- coding: utf-8 -*-
# @Time    : 2021/1/4
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
from collections import OrderedDict

import numpy as np
import py_sod_metrics


def ndarray_to_basetype(data):
    """
    将单独的ndarray，或者tuple，list或者dict中的ndarray转化为基本数据类型，
    即列表(.tolist())和python标量
    """

    def _to_list_or_scalar(item):
        listed_item = item.tolist()
        if isinstance(listed_item, list) and len(listed_item) == 1:
            listed_item = listed_item[0]
        return listed_item

    if isinstance(data, (tuple, list)):
        results = [_to_list_or_scalar(item) for item in data]
    elif isinstance(data, dict):
        results = {k: _to_list_or_scalar(item) for k, item in data.items()}
    else:
        assert isinstance(data, np.ndarray)
        results = _to_list_or_scalar(data)
    return results


def round_w_zero_padding(x, bit_width):
    x = str(round(x, bit_width))
    x += "0" * (bit_width - len(x.split(".")[-1]))
    return x


INDIVADUAL_METRIC_MAPPING = {
    "mae": py_sod_metrics.MAE,
    # "fm": py_sod_metrics.Fmeasure,
    "em": py_sod_metrics.Emeasure,
    "sm": py_sod_metrics.Smeasure,
    "wfm": py_sod_metrics.WeightedFmeasure,
}

BINARY_METRIC_MAPPING = {
    # gray-scale
    "fmeasure": {
        "handler": py_sod_metrics.FmeasureHandler,
        "kwargs": dict(with_dynamic=True, with_adaptive=True, beta=0.3),
    },
    "precision": {
        "handler": py_sod_metrics.PrecisionHandler,
        "kwargs": dict(with_dynamic=True, with_adaptive=False),
    },
    "recall": {
        "handler": py_sod_metrics.RecallHandler,
        "kwargs": dict(with_dynamic=True, with_adaptive=False),
    },
    "iou": {
        "handler": py_sod_metrics.IOUHandler,
        "kwargs": dict(with_dynamic=True, with_adaptive=True),
    },
    "dice": {
        "handler": py_sod_metrics.DICEHandler,
        "kwargs": dict(with_dynamic=True, with_adaptive=True),
    },
    "specificity": {
        "handler": py_sod_metrics.SpecificityHandler,
        "kwargs": dict(with_dynamic=True, with_adaptive=True),
    },
    "ber": {
        "handler": py_sod_metrics.BERHandler,
        "kwargs": dict(with_dynamic=True, with_adaptive=True),
    },
    # binary metrics average over the each sample
    "bifmeasure": {
        "handler": py_sod_metrics.FmeasureHandler,
        "kwargs": dict(
            with_dynamic=False, with_adaptive=False, with_binary=True, sample_based=True, beta=1
        ),
    },
    "biprecision": {
        "handler": py_sod_metrics.PrecisionHandler,
        "kwargs": dict(
            with_dynamic=False, with_adaptive=False, with_binary=True, sample_based=True
        ),
    },
    "birecall": {
        "handler": py_sod_metrics.RecallHandler,
        "kwargs": dict(
            with_dynamic=False, with_adaptive=False, with_binary=True, sample_based=True
        ),
    },
    "biiou": {
        "handler": py_sod_metrics.IOUHandler,
        "kwargs": dict(
            with_dynamic=False, with_adaptive=False, with_binary=True, sample_based=True
        ),
    },
    "bidice": {
        "handler": py_sod_metrics.DICEHandler,
        "kwargs": dict(
            with_dynamic=False, with_adaptive=False, with_binary=True, sample_based=True
        ),
    },
    "bispecificity": {
        "handler": py_sod_metrics.SpecificityHandler,
        "kwargs": dict(
            with_dynamic=False, with_adaptive=False, with_binary=True, sample_based=True
        ),
    },
    "biber": {
        "handler": py_sod_metrics.BERHandler,
        "kwargs": dict(
            with_dynamic=False, with_adaptive=False, with_binary=True, sample_based=True
        ),
    },
}

GRAYSCALE_METRICS = ["em"] + [k for k in BINARY_METRIC_MAPPING.keys() if not k.startswith("bi")]

SUPPORTED_METRICS = ["mae", "em", "sm", "wfm"] + sorted(BINARY_METRIC_MAPPING.keys())


class GrayscaleMetricRecorder:
    # 'fm' is replaced by 'fmeasure' in BINARY_METRIC_MAPPING
    suppoted_metrics = ["mae", "em", "sm", "wfm"] + sorted(
        [k for k in BINARY_METRIC_MAPPING.keys() if not k.startswith("bi")]
    )

    def __init__(self, metric_names=None):
        """
        用于统计各种指标的类
        """
        if not metric_names:
            metric_names = self.suppoted_metrics
        assert all(
            [m in self.suppoted_metrics for m in metric_names]
        ), f"Only support: {self.suppoted_metrics}"

        self.metric_objs = {}
        has_existed = False
        for metric_name in metric_names:
            if metric_name in INDIVADUAL_METRIC_MAPPING:
                self.metric_objs[metric_name] = INDIVADUAL_METRIC_MAPPING[metric_name]()
            else:  # metric_name in BINARY_METRIC_MAPPING
                if not has_existed:  # only init once
                    self.metric_objs["fmeasurev2"] = py_sod_metrics.FmeasureV2()
                    has_existed = True
                metric_handler = BINARY_METRIC_MAPPING[metric_name]
                self.metric_objs["fmeasurev2"].add_handler(
                    handler_name=metric_name,
                    metric_handler=metric_handler["handler"](**metric_handler["kwargs"]),
                )

    def step(self, pre: np.ndarray, gt: np.ndarray, gt_path: str):
        assert pre.shape == gt.shape, (pre.shape, gt.shape, gt_path)
        assert pre.dtype == gt.dtype == np.uint8, (pre.dtype, gt.dtype, gt_path)

        for m_obj in self.metric_objs.values():
            m_obj.step(pre, gt)

    def show(self, num_bits: int = 3, return_ndarray: bool = False) -> dict:
        """
        返回指标计算结果：

        - 曲线数据(sequential)
        - 数值指标(numerical)
        """
        sequential_results = {}
        numerical_results = {}
        for m_name, m_obj in self.metric_objs.items():
            info = m_obj.get_results()
            if m_name == "fmeasurev2":
                for _name, results in info.items():
                    dynamic_results = results.get("dynamic")
                    adaptive_results = results.get("adaptive")
                    if dynamic_results is not None:
                        sequential_results[_name] = np.flip(dynamic_results)
                        numerical_results[f"max{_name}"] = dynamic_results.max()
                        numerical_results[f"avg{_name}"] = dynamic_results.mean()
                    if adaptive_results is not None:
                        numerical_results[f"adp{_name}"] = adaptive_results
            else:
                results = info[m_name]
                if m_name in ("wfm", "sm", "mae"):
                    numerical_results[m_name] = results
                elif m_name == "em":
                    sequential_results[m_name] = np.flip(results["curve"])
                    numerical_results.update(
                        {
                            "maxem": results["curve"].max(),
                            "avgem": results["curve"].mean(),
                            "adpem": results["adp"],
                        }
                    )
                else:
                    raise NotImplementedError(m_name)

        if num_bits is not None and isinstance(num_bits, int):
            numerical_results = {k: v.round(num_bits) for k, v in numerical_results.items()}
        if not return_ndarray:
            sequential_results = ndarray_to_basetype(sequential_results)
            numerical_results = ndarray_to_basetype(numerical_results)
        return {"sequential": sequential_results, "numerical": numerical_results}


class BinaryMetricRecorder:
    suppoted_metrics = ["mae", "sm", "wfm"] + sorted(
        [k for k in BINARY_METRIC_MAPPING.keys() if k.startswith("bi")]
    )

    def __init__(
        self, metric_names=("bif1", "biprecision", "birecall", "biiou", "bioa", "bikappa")
    ):
        """
        用于统计各种指标的类
        """
        if not metric_names:
            metric_names = self.suppoted_metrics
        assert all(
            [m in self.suppoted_metrics for m in metric_names]
        ), f"Only support: {self.suppoted_metrics}"

        self.metric_objs = {}
        has_existed = False
        for metric_name in metric_names:
            if metric_name in INDIVADUAL_METRIC_MAPPING:
                self.metric_objs[metric_name] = INDIVADUAL_METRIC_MAPPING[metric_name]()
            else:  # metric_name in BINARY_METRIC_MAPPING
                if not has_existed:  # only init once
                    self.metric_objs["fmeasurev2"] = py_sod_metrics.FmeasureV2()
                    has_existed = True
                metric_handler = BINARY_METRIC_MAPPING[metric_name]
                self.metric_objs["fmeasurev2"].add_handler(
                    handler_name=metric_name,
                    metric_handler=metric_handler["handler"](**metric_handler["kwargs"]),
                )

    def step(self, pre: np.ndarray, gt: np.ndarray, gt_path: str):
        assert pre.shape == gt.shape, (pre.shape, gt.shape, gt_path)
        assert pre.dtype == gt.dtype == np.uint8, (pre.dtype, gt.dtype, gt_path)

        for m_obj in self.metric_objs.values():
            m_obj.step(pre, gt)

    def show(self, num_bits: int = 3, return_ndarray: bool = False) -> dict:
        numerical_results = {}
        for m_name, m_obj in self.metric_objs.items():
            info = m_obj.get_results()
            if m_name == "fmeasurev2":
                for _name, results in info.items():
                    binary_results = results.get("binary")
                    if binary_results is not None:
                        numerical_results[_name] = binary_results
            else:
                results = info[m_name]
                if m_name in ("wfm", "sm", "mae"):
                    numerical_results[m_name] = results
                else:
                    raise NotImplementedError(m_name)

        if num_bits is not None and isinstance(num_bits, int):
            numerical_results = {k: v.round(num_bits) for k, v in numerical_results.items()}
        if not return_ndarray:
            numerical_results = ndarray_to_basetype(numerical_results)
        return {"numerical": numerical_results}


class GroupedMetricRecorder:
    def __init__(
        self, group_names=None, metric_names=("sm", "wfm", "mae", "fmeasure", "em", "iou", "dice")
    ):
        self.group_names = group_names
        self.metric_names = metric_names
        self.zero()

    def zero(self):
        self.metric_recorders = {}
        if self.group_names is not None:
            self.metric_recorders.update(
                {
                    n: GrayscaleMetricRecorder(metric_names=self.metric_names)
                    for n in self.group_names
                }
            )

    def step(self, group_name: str, pre: np.ndarray, gt: np.ndarray):
        if group_name not in self.metric_recorders:
            self.metric_recorders[group_name] = GrayscaleMetricRecorder(
                metric_names=self.metric_names
            )
        self.metric_recorders[group_name].step(pre, gt)

    def show(self, num_bits: int = 3, return_group: bool = False):
        groups_metrics = {
            n: r.show(num_bits=None, return_ndarray=True) for n, r in self.metric_recorders.items()
        }

        results = {}
        for group_metrics in groups_metrics.values():
            for (
                metric_group_name,
                metric_group,
            ) in group_metrics.items():  # sequential and numerical
                for metric_name, metric_array in metric_group.items():
                    results.setdefault(metric_group_name, {}).setdefault(metric_name, []).append(
                        metric_array
                    )

        numerical_results = {}
        sequential_results = {}
        for metric_group_name, metric_group in results.items():
            for metric_name, metric_array in metric_group.items():
                metric_array = np.mean(np.vstack(metric_array), axis=0)  # average over all groups

                if metric_name in BINARY_METRIC_MAPPING or metric_name == "em":
                    if metric_group_name == "sequential":
                        numerical_results[f"max{metric_name}"] = metric_array.max()
                        numerical_results[f"avg{metric_name}"] = metric_array.mean()
                        sequential_results[metric_name] = metric_array
                else:
                    if metric_group_name == "numerical":
                        if metric_name.startswith(("max", "avg")):
                            # these metrics (maxfm, avgfm, maxem, avgem) will be recomputed within the group
                            continue
                        numerical_results[metric_name] = metric_array

        numerical_results = ndarray_to_basetype(numerical_results)
        numerical_results = {
            name: round_w_zero_padding(metric, bit_width=num_bits)
            for name, metric in numerical_results.items()
        }
        numerical_results = self.sort_results(numerical_results)

        sequential_results = ndarray_to_basetype(sequential_results)

        if return_group:
            group_numerical_results = {}
            for group_name, group_metric in groups_metrics.items():
                group_metric = {k: v.round(num_bits) for k, v in group_metric["numerical"].items()}
                group_metric = ndarray_to_basetype(group_metric)
                group_numerical_results[group_name] = self.sort_results(group_metric)

            return {"sequential": sequential_results, "numerical": group_numerical_results}
        return {"sequential": sequential_results, "numerical": numerical_results}

    def sort_results(self, results: dict) -> OrderedDict:
        """for a single group of metrics"""
        sorted_results = OrderedDict()
        a = "abcd"
        all_keys = sorted(results.keys(), key=lambda item: item[::-1])
        for name in self.metric_names:
            for key in all_keys:
                if key.endswith(name):
                    sorted_results[key] = results[key]
        return sorted_results
