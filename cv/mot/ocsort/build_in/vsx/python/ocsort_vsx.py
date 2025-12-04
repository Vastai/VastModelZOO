# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from loguru import logger
import cv2
import glob
import os

import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../..')

from source_code.trackers.ocsort_tracker.ocsort import OCSort
from source_code.trackers.tracking_utils.timer import Timer

from source_code.detect_vsx import Detector

import motmetrics as mm
import argparse
import os
import time
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--path",default="/path/to/mot/train/",help="path to test images")
    # detect args
    parser.add_argument("--model_prefix_path", type=str, default="deploy_weights/pytorch_ocsort_run_stream_int8/mod", help="model info")
    parser.add_argument("--vdsp_params_info", type=str, default="../build_in/vdsp_params/pytorch-ocsort_mot17_ablation-vdsp_params.json", help="vdsp op info",)
    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument("--batch", type=int, default=1, help="bacth size")
    # tracking args
    parser.add_argument("--track_thresh",type=float,default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer",type=int,default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh",type=int,default=0.9, help="matching threshold for tracking")
    parser.add_argument('--min-box-area',type=float,default=100, help='filter out tiny boxes')
    parser.add_argument('--result_dir',type=str,default="result/track_eval/", help='filter out tiny boxes')
    return parser


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info('Comparing {}...'.format(k))
            accs.append(
                mm.utils.compare_to_groundtruth(gts[k],
                                                tsacc,
                                                'iou',
                                                distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id,
                                          id=track_id,
                                          x1=round(x1, 1),
                                          y1=round(y1, 1),
                                          w=round(w, 1),
                                          h=round(h, 1),
                                          s=1)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def track(test_mot_dataset, predict, vis_folder, current_time, detect_size,
          args):
    for seq in test_mot_dataset:
        print(seq)
        if 'MOT17-05-FRCNN' in seq or 'MOT17-06-FRCNN' in seq:
            args.track_buffer = 14
        elif 'MOT17-13-FRCNN' in seq or 'MOT17-14-FRCNN' in seq:
            args.track_buffer = 25
        else:
            args.track_buffer = 30

        img_list = sorted(glob.glob(os.path.join(seq, 'img1/*')),
                          key=lambda name: int(name[-10:-4]))
        # tracker = BYTETracker(args)
        tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=0.3, use_byte=False)
        results = []

        # for frame_id, im in tqdm(enumerate(img_list)):
        for frame_id, im in tqdm(enumerate(img_list), 
                            desc="Processing images",
                            unit="frame",
                            total=len(img_list)):
            # print(im)
            if frame_id + 1 == 1:
                # tracker = BYTETracker(args)
                tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=0.3, use_byte=False)
                if len(results) != 0:
                    result_filename = os.path.join(vis_folder, '{}.txt'.format(seq.split('/')[-1]))
                    write_results(result_filename, results)
                    results = []

            img = cv2.imread(im)
            outputs = predict.run_sync(img)
            # print(outputs)

            # exit(0)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(
                    outputs, [img.shape[0], img.shape[1]],
                    detect_size)
                online_tlwhs = []
                online_ids = []
                # online_scores = []
                for t in online_targets:
                    # tlwh = t.tlwh
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    # print(t)
                    # tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        # online_scores.append(float(t[5]))
                # save results
                results.append(
                    (frame_id + 1, online_tlwhs, online_ids))

        result_filename = os.path.join(vis_folder,
                                       '{}.txt'.format(seq.split('/')[-1]))
        write_results(result_filename, results)

    # evaluate MOTA
    mm.lap.default_solver = 'lap'

    gt_type = ''
    print('gt_type', gt_type)
    gtfiles = glob.glob(
        os.path.join(args.path, '*/gt/gt{}.txt'.format(gt_type)))
    print('gt_files', gtfiles)
    tsfiles = [
        f for f in glob.glob(os.path.join(vis_folder, '*.txt'))
        if not os.path.basename(f).startswith('eval')
    ]

    logger.info('Found {} groundtruths and {} test files.'.format(
        len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')

    gt = OrderedDict([(Path(f).parts[-3],
                       mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1))
                      for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0],
                       mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1))
                      for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    logger.info('Running metrics')
    metrics = [
        'recall', 'precision', 'num_unique_objects', 'mostly_tracked',
        'partially_tracked', 'mostly_lost', 'num_false_positives',
        'num_misses', 'num_switches', 'num_fragmentations', 'mota', 'motp',
        'num_objects'
    ]
    summary = mh.compute_many(accs,
                              names=names,
                              metrics=metrics,
                              generate_overall=True)
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    # print(mm.io.render_summary(
    #   summary, formatters=mh.formatters,
    #   namemap=mm.io.motchallenge_metric_names))
    div_dict = {
        'num_objects': [
            'num_false_positives', 'num_misses', 'num_switches',
            'num_fragmentations'
        ],
        'num_unique_objects':
        ['mostly_tracked', 'partially_tracked', 'mostly_lost']
    }
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = [
        'num_false_positives', 'num_misses', 'num_switches',
        'num_fragmentations', 'mostly_tracked', 'partially_tracked',
        'mostly_lost'
    ]
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(
        mm.io.render_summary(summary,
                             formatters=fmt,
                             namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs,
                              names=names,
                              metrics=metrics,
                              generate_overall=True)
    print(
        mm.io.render_summary(summary,
                             formatters=mh.formatters,
                             namemap=mm.io.motchallenge_metric_names))
    logger.info('Completed')


def main(args):
    vis_folder = os.path.join(args.result_dir)
    os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    # test_mot_dataset = glob.glob(args.path + '*')
    # test_mot_dataset = [f for f in test_mot_dataset if 'FRCNN' in f]
    test_mot_dataset = [
        'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN',
        'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'
    ]
    test_mot_dataset = [os.path.join(args.path, f) for f in test_mot_dataset]

    # Load model
    detect_size = (800, 1440)
    predict = Detector(model_prefix_path=args.model_prefix_path,
                        vdsp_params_info=args.vdsp_params_info,
                        device_id=args.device_id,
                        batch_size=args.batch,
                        balance_mode = 0,
                        is_async_infer = False,
                        model_output_op_name = "", 
                    )

    current_time = time.localtime()
    track(test_mot_dataset, predict, vis_folder, current_time, detect_size,
          args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
