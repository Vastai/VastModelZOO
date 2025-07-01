import numpy as np
import sys
import re
import os
import string
import argparse

def normalize_text(text):
    text = ''.join(
        filter(lambda x: x in (string.digits + string.ascii_letters), text))
    return text.lower()

def read_label(dataset_label_file):
    y = open(dataset_label_file)
    lines = y.readlines()
    batchs = {}
    for line in lines:
        line = line.replace('\n','').split(' ')
        batchs[line[0].split('/')[-1]] = line[1]
    return batchs

if __name__ == '__main__':

    parse = argparse.ArgumentParser(description="TEST CRNN acc")
    parse.add_argument("--gt_file", type=str, default="", help="gt")
    parse.add_argument("--output_file", type=str, default="", help="pred")
    args = parse.parse_args()

    gts = read_label(args.gt_file)
    outputs = read_label(args.output_file)
    outputs = dict(sorted(outputs.items()))

    right_num = 0
    for key in outputs.keys():
        print(f"key-{key}, outputs[key]-{outputs[key]}, gts[key]-{gts[key]}")
        if normalize_text(outputs[key]) == normalize_text(gts[key]):
            right_num += 1
        else:
            print(f"=====>>>>> ERROR: key-{key}, outputs[key]-{outputs[key]}, gts[key]-{gts[key]}")
    print(f'right_num = {right_num} all_num={len(outputs)}, acc = {right_num/len(outputs)}')


