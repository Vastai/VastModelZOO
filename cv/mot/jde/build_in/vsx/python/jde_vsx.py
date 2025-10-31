# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm

import torch

import sys
_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../..')

from source_code.tracker.tracker import JDETracker
from source_code.utils import visualization as vis
from source_code.utils.log import logger
from source_code.utils.timer import Timer
from source_code.utils.evaluation import Evaluator
from source_code.utils.parse_config import parse_model_cfg
import source_code.utils.datasets as datasets
from source_code.utils.utils import *


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    '''
       Processes the video sequence given and provides the output of tracking result (write the results in video file)

       It uses JDE model for getting information about the online targets present.

       Parameters
       ----------
       opt : Namespace
             Contains information passed as commandline arguments.

       dataloader : LoadVideo
                    Instance of LoadVideo class used for fetching the image sequence and associated data.

       data_type : String
                   Type of dataset corresponding(similar) to the given video.

       result_filename : String
                         The name(path) of the file for storing results.

       save_dir : String
                  Path to the folder for storing the frames containing bounding box information (Result frames).

       show_image : bool
                    Option for shhowing individial frames during run-time.

       frame_rate : int
                    Frame-rate of the given video.

       Returns
       -------
       (Returns are not significant here)
       frame_id : int
                  Sequence number of the last sequence
       '''

    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, 'deploy_weights/jde_1088x608_uncertainty-int8-max-1_3_608_1088-vacc/jde_1088x608_uncertainty', 'vdsp_params.json', frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for path, img, img0 in dataloader:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1./max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0, path)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), 
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results')
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..','outputs', seq) if save_images or save_videos else None

        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), img_size=(1088, 608))
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read() 
        frame_rate = int(meta_info[meta_info.find('frameRate')+10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track.py')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--test-mot16', action='store_true', help='tracking buffer')
    parser.add_argument('--save-images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save-videos', action='store_true', help='save tracking results (video)')
    opt = parser.parse_args()
    print(opt, end='\n\n')
 
    if not opt.test_mot16:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP
                    '''
        data_root = './datasets/tracking/MOT17/MOT17/train/'
    else:
        # seqs_str = '''MOT16-01
        #              MOT16-03
        #              MOT16-06
        #              MOT16-07
        #              MOT16-08
        #              MOT16-12
        #              MOT16-14'''
        seqs_str = '''MOT16-02
                     MOT16-04
                     MOT16-05
                     MOT16-09
                     MOT16-10
                     MOT16-11
                     MOT16-13'''
        # seqs_str = '''MOT16-02'''
        data_root = './datasets/tracking/MOT16/train'
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         show_image=False,
         save_images=opt.save_images, 
         save_videos=opt.save_videos)
    
'''
int8 max
              IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT ML   FP    FN  IDs    FM  MOTA  MOTP IDt IDa IDm
MOT17-02-SDP 50.4% 57.5% 44.8% 71.6% 92.1%  62  22  32  8 1147  5276  307   636 63.8% 0.218 134  76   8
MOT17-04-SDP 74.6% 82.3% 68.2% 78.6% 94.8%  83  44  24 15 2035 10182  138   467 74.0% 0.192  51  43   3
MOT17-05-SDP 77.0% 82.7% 72.0% 83.3% 95.8% 133  75  49  9  253  1152  154   212 77.5% 0.202  87  55  23
MOT17-09-SDP 61.0% 67.1% 56.0% 78.6% 94.2%  26  16  10  0  258  1137  116   147 71.6% 0.179  62  34   9
MOT17-10-SDP 64.8% 70.5% 59.9% 79.2% 93.1%  57  28  28  1  754  2673  295   565 71.0% 0.221 117  92   7
MOT17-11-SDP 78.5% 81.4% 75.8% 89.6% 96.1%  75  50  21  4  340   986   91   206 85.0% 0.170  36  35   4
MOT17-13-SDP 71.9% 79.4% 65.7% 77.6% 93.8% 110  60  42  8  598  2608  246   579 70.3% 0.223 123  80  37
OVERALL      69.1% 76.0% 63.4% 78.6% 94.3% 546 295 206 45 5385 24014 1347  2812 72.6% 0.200 610 415  91
'''