# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import random

from nltk.metrics.distance import edit_distance


class Accuracy(object):

    def __init__(self):
        self.reset()
        self.predict_example_log = None

    @property
    def result(self):
        res = {
            'acc': self.avg['acc']['true'],
            'edit_distance': self.avg['edit'],
        }
        return res

    def measure(self, preds, preds_prob, gts):
        batch_size = len(gts)
        true_num = 0
        norm_ED = 0
        for pstr, gstr in zip(preds, gts):
            if pstr == gstr:
                true_num += 1

            if len(pstr) == 0 or len(gstr) == 0:
                norm_ED += 0
            elif len(gstr) > len(pstr):
                norm_ED += 1 - edit_distance(pstr, gstr) / len(gstr)
            else:
                norm_ED += 1 - edit_distance(pstr, gstr) / len(pstr)
        if preds_prob is not None:
            self.show_example(preds, preds_prob, gts)
        self.all['acc']['true'] += true_num
        self.all['acc']['false'] += (batch_size - true_num)
        self.all['edit'] += norm_ED
        self.count += batch_size
        for key, value in self.all['acc'].items():
            self.avg['acc'][key] = self.all['acc'][key] / self.count
        self.avg['edit'] = self.all['edit'] / self.count

    def reset(self):
        self.all = dict(
            acc=dict(
                true=0,
                false=0
            ),
            edit=0
        )
        self.avg = dict(
            acc=dict(
                true=0,
                false=0
            ),
            edit=0
        )
        self.count = 0

    def show_example(self, preds, preds_prob, gts):
        count = 0
        self.predict_example_log = None
        dashed_line = '-' * 80
        head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
        self.predict_example_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
        # for gt, pred, prob in zip(gts[:5], preds[:5], preds_prob[:5]):
        # show_inds = random.choices(range(len(gts)), k=5)
        show_inds = list(range(len(gts)))
        random.shuffle(show_inds)
        show_inds = show_inds[:5]
        show_gts = [gts[i] for i in show_inds]
        show_preds = [preds[i] for i in show_inds]
        show_prob = [preds_prob[i] for i in show_inds]
        for gt, pred, prob in zip(show_gts, show_preds, show_prob):
            self.predict_example_log += f'{gt:25s} | {pred:25s} | {prob:0.4f}\t{str(pred == gt)}\n'
            count += 1
            if count > 4:
                break
        self.predict_example_log += f'{dashed_line}'

        return self.predict_example_log


def get_text(text_path,prob_need = False):
    info_dict = {}
    with open(text_path)as fout:
        lines = fout.readlines()
        for i in range(0, len(lines)):
            file_name = lines[i].split(" ")[0]
            str = lines[i].split(" ")[1].lower().strip()
            if prob_need:
                prob = lines[i].split(" ")[2].strip()
                info_dict[file_name] = [[str],[float(prob)]]
            else:
                info_dict[file_name] = [str]
    return info_dict

if __name__ == '__main__':
    pred_txt = "../../workspace/weights/decode.txt"
    gt_txt = "../../workspace/weights/test_info.txt"

    metric = Accuracy()
    metric.reset()
    pred_dict = get_text(pred_txt,prob_need = True)
    gt_dict = get_text(gt_txt)

    for file in pred_dict.keys():
        metric.measure(pred_dict[file][0], pred_dict[file][1], gt_dict[file])
    print('Test, average acc %.4f, edit distance %s' % (metric.avg['acc']['true'],
                                                                       metric.avg['edit']))
