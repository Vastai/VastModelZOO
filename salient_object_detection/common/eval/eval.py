# -*- coding: utf-8 -*-
import argparse
import os
import textwrap
import warnings

from metrics import cal_sod_matrics
from utils.generate_info import get_datasets_info, get_methods_info
from utils.recorders import SUPPORTED_METRICS


def get_args():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            r"""
    A Powerful Evaluation Toolkit based on PySODMetrics.

    INCLUDE: More metrics can be set in `utils/recorders/metric_recorder.py`

    - F-measure-Threshold Curve
    - Precision-Recall Curve
    - MAE
    - weighted F-measure
    - S-measure
    - max/average/adaptive/binary F-measure
    - max/average/adaptive/binary E-measure
    - max/average/adaptive/binary Precision
    - max/average/adaptive/binary Recall
    - max/average/adaptive/binary Sensitivity
    - max/average/adaptive/binary Specificity
    - max/average/adaptive/binary F-measure
    - max/average/adaptive/binary Dice
    - max/average/adaptive/binary IoU

    NOTE:

    - Our method automatically calculates the intersection of `pre` and `gt`.
    - Currently supported pre naming rules: `prefix + gt_name_wo_ext + suffix_w_ext`

    EXAMPLES:

    python eval_image.py \
        --dataset-json configs/datasets/rgbd_sod.json \
        --method-json \
            configs/methods/json/rgbd_other_methods.json \
            configs/methods/json/rgbd_our_method.json \
        --metric-names sm wfm mae fmeasure em \
        --num-bits 4 \
        --num-workers 4 \
        --metric-npy output/rgbd_metrics.npy \
        --curves-npy output/rgbd_curves.npy \
        --record-txt output/rgbd_results.txt
        --to-overwrite \
        --record-xlsx output/test-metric.xlsx \
        --include-dataset \
            dataset-name1-from-dataset-json \
            dataset-name2-from-dataset-json \
            dataset-name3-from-dataset-json
        --include-methods \
            method-name1-from-method-json \
            method-name2-from-method-json \
            method-name3-from-method-json
    """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset-json", required=False, type=str, default="salient_object_detection/common/eval/examples/config_dataset.json", help="Json file for datasets.")
    parser.add_argument(
        "--method-json", required=False, nargs="+", type=str, default="salient_object_detection/common/eval/examples/config_method.json", help="Json file for methods."
    )
    parser.add_argument("--metric-npy", type=str, help="Npy file for saving metric results.")
    parser.add_argument("--curves-npy", type=str, help="Npy file for saving curve results.")
    parser.add_argument("--record-txt", type=str, help="Txt file for saving metric results.")
    parser.add_argument("--to-overwrite", action="store_true", help="To overwrite the txt file.")
    parser.add_argument("--record-xlsx", type=str, help="Xlsx file for saving metric results.")
    parser.add_argument(
        "--include-methods",
        type=str,
        nargs="+",
        help="Names of only specific methods you want to evaluate.",
    )
    parser.add_argument(
        "--exclude-methods",
        type=str,
        nargs="+",
        help="Names of some specific methods you do not want to evaluate.",
    )
    parser.add_argument(
        "--include-datasets",
        type=str,
        nargs="+",
        help="Names of only specific datasets you want to evaluate.",
    )
    parser.add_argument(
        "--exclude-datasets",
        type=str,
        nargs="+",
        help="Names of some specific datasets you do not want to evaluate.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for multi-threading or multi-processing. Default: 4",
    )
    parser.add_argument(
        "--num-bits",
        type=int,
        default=3,
        help="Number of decimal places for showing results. Default: 3",
    )
    parser.add_argument(
        "--metric-names",
        type=str,
        nargs="+",
        default=["mae", "fmeasure", "precision", "recall", "em", "sm", "wfm"],
        choices=SUPPORTED_METRICS,
        help="Names of metrics",
    )
    args = parser.parse_args()

    if args.metric_npy:
        os.makedirs(os.path.dirname(args.metric_npy), exist_ok=True)
    if args.curves_npy:
        os.makedirs(os.path.dirname(args.curves_npy), exist_ok=True)
    if args.record_txt:
        os.makedirs(os.path.dirname(args.record_txt), exist_ok=True)
    if args.record_xlsx:
        os.makedirs(os.path.dirname(args.record_xlsx), exist_ok=True)
    if args.to_overwrite and not args.record_txt:
        warnings.warn("--to-overwrite only works with a valid --record-txt")
    return args


def main():
    args = get_args()

    # 包含所有数据集信息的字典
    datasets_info = get_datasets_info(
        datastes_info_json=args.dataset_json,
        include_datasets=args.include_datasets,
        exclude_datasets=args.exclude_datasets,
    )
    # 包含所有待比较模型结果的信息的字典
    methods_info = get_methods_info(
        methods_info_jsons=args.method_json,
        for_drawing=True,
        include_methods=args.include_methods,
        exclude_methods=args.exclude_methods,
    )

    # 确保多进程在windows上也可以正常使用
    cal_sod_matrics.cal_image_matrics(
        sheet_name="Results",
        to_append=not args.to_overwrite,
        txt_path=args.record_txt,
        xlsx_path=args.record_xlsx,
        methods_info=methods_info,
        datasets_info=datasets_info,
        curves_npy_path=args.curves_npy,
        metrics_npy_path=args.metric_npy,
        num_bits=args.num_bits,
        num_workers=args.num_workers,
        metric_names=args.metric_names,
        ncols_tqdm=119,
    )


if __name__ == "__main__":
    main()
