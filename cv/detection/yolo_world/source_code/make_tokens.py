import numpy as np

from transformers import AutoTokenizer
import json
import os
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class_text",
        default="../data/labels/lvis_v1_class_texts.json",
        help="class label file",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="/opt/vastai/vaststreamx/data/tokernizer/clip-vit-base-patch32",
        help="tokenizer path",
    )
    parser.add_argument(
        "--save_path",
        default="tokens",
        help="path to save tokens",
    )
    args = parser.parse_args()
    return args


def make_tokens(tokenizer, text):
    assert isinstance(text, str), f"input type must be str array"
    token_dict = tokenizer(text=text, return_tensors="pt", padding=True)
    token = token_dict["input_ids"][0]
    input_seq_len = 16
    token_padding = np.full([input_seq_len], 49407, dtype=np.int32)  # pad
    token_padding[: len(token)] = token
    # make mask
    token_mask = np.ones(shape=(input_seq_len), dtype=np.int32) * (-1)
    mask = token_dict["attention_mask"][0]
    token_mask[: len(mask)] = mask
    # make input
    zero_arr = np.zeros(token_padding.shape, dtype=np.int32)
    tokens = []
    tokens.append(token_padding)
    tokens.append(zero_arr)
    tokens.append(zero_arr)
    tokens.append(token_mask)
    tokens.append(zero_arr)
    tokens.append(zero_arr)

    return tokens


if __name__ == "__main__":

    args = argument_parser()

    class_text = args.class_text
    tokenizer_path = args.tokenizer_path
    save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load text classes
    with open(class_text) as f:
        text_classes = json.load(f)
    text_classes = [x[0] for x in text_classes]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    for index, text in enumerate(text_classes):
        print(text)
        tokens = make_tokens(tokenizer, text)
        out = {}
        out["input_0"] = tokens[0]
        out["input_1"] = tokens[1]
        out["input_2"] = tokens[2]
        out["input_3"] = tokens[3]
        out["input_4"] = tokens[4]
        out["input_5"] = tokens[5]
        save_file = os.path.join(save_path, text + ".npz")
        np.savez(save_file, **out)
        print(f"save result to: {save_file}")
