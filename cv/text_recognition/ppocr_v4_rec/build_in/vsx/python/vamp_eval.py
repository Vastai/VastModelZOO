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

class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)

        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

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
    parse.add_argument("--datasets_name", type=str, default="IIIT5k_3000")
    parse.add_argument("--label_file", type=str, default="./eval/IIIT5k_3000.csv", help="gt")
    parse.add_argument("--pred_npz_dir", type=str, default="./eval/model_latency_npz", help="pred")
    parse.add_argument("--npz_datalist", type=str, default="datalist.txt", help="pred")
    args = parse.parse_args()

    converter = CTCLabelDecode()

    with open(args.npz_datalist) as f:
        datalist = f.readlines()

    labels = read_label(args.label_file)
    right_num = 0
    for index in range(len(labels)):
        file_name = datalist[index].replace('\n','').split('/')[-1].split('.')[0]
        pred = np.load(os.path.join(args.pred_npz_dir, 'output_' + str(index).zfill(6) + ".npz"))
        pred_str = converter(pred['output_0'].reshape(1,25,37))
        print(pred_str)
        if normalize_text(pred_str[0][0]) == normalize_text(labels[file_name]):
            right_num += 1
    print(args.datasets_name, 'right_num, all_num, acc = ', right_num, len(labels), right_num/len(labels))


