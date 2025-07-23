
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

import torch
from transformers import AutoTokenizer
import numpy as np
import argparse


import common.utils as utils


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path",
        default="/path/to/tokenizer/bert-base-uncased",
        help="tokenizer path",
    )
    parser.add_argument(
        "--label_file",
        default="/path/to/coco91.txt",
        help="label file",
    )
    parser.add_argument(
        "--output_file",
        default="tokens.npz",
        help="output file",
    )
    args = parser.parse_args()
    return args


def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list):
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token, device=input_ids.device)
        .bool()
        .unsqueeze(0)
        .repeat(bs, 1, 1)
    )
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[
                row, previous_col + 1 : col + 1, previous_col + 1 : col + 1
            ] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )
            c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
            c2t_maski[previous_col + 1 : col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col

    cate_to_token_mask_list = [
        torch.stack(cate_to_token_mask_listi, dim=0)
        for cate_to_token_mask_listi in cate_to_token_mask_list
    ]
    return attention_mask, position_ids.to(torch.long)
    # return attention_mask, position_ids.to(torch.long), cate_to_token_mask_list


def text_tokenization(tokenizer, specical_tokens, text):
    assert isinstance(text, str), f"input type must be str"
    token_dict = tokenizer(text=text, return_tensors="pt", padding="longest")
    token = token_dict["input_ids"][0]
    input_len = 208  # model_input_len
    seq_len = len(token)
    assert (
        seq_len <= input_len
    ), f"token len=({seq_len}) is larger than model max len=({input_len}),please input shorter string"

    # input_ids
    input_ids = np.full([1, input_len], 0, dtype=np.int32)  # pad
    input_ids[0, : len(token)] = token

    # make mask
    token_mask, text_position_ids = generate_masks_with_special_tokens_and_transfer_map(
        token_dict, specical_tokens
    )
    token_mask = token_mask.numpy()
    text_position_ids = text_position_ids.numpy()

    # position_ids
    position_ids = np.full([1, input_len], 0, dtype=np.int32)  # pad
    position_ids[:, :seq_len] = text_position_ids

    # token_type_ids
    token_type_ids = np.zeros((1, input_len), dtype=np.int32)
    token_type_ids[0, :seq_len] = token_dict["token_type_ids"][0]

    # attention_mask
    text_attention_mask = token_dict["attention_mask"].numpy()
    attention_mask = np.zeros(input_ids.shape, dtype=np.int32)
    attention_mask[0, :195] = text_attention_mask[0, :195]

    # text_token_mask
    text_token_mask = np.zeros((208, 208), dtype=np.int32)
    text_token_mask[:195, :195] = token_mask.astype(np.int32)
    text_token_mask[text_token_mask == 0] = -10000
    text_token_mask[text_token_mask == 1] = 0
    text_token_mask = text_token_mask.astype(np.float16)

    # make input
    tokens = []
    tokens.append(input_ids)
    tokens.append(position_ids)
    tokens.append(token_type_ids)
    tokens.append([])
    tokens.append(text_token_mask)
    tokens.append([])

    attention_mask_for_decoder = attention_mask.astype(np.float16)
    attention_mask_for_decoder[attention_mask_for_decoder == 0.0] = float("-inf")
    attention_mask_for_decoder[attention_mask_for_decoder == 1.0] = 0.0

    text_token_mask_for_decoder = np.zeros((208, 208), dtype=np.int32)
    text_token_mask_for_decoder[:195, :195] = token_mask.astype(np.int32)
    text_token_mask_for_decoder = text_token_mask_for_decoder.astype(np.float16)

    return [tokens, attention_mask_for_decoder, text_token_mask_for_decoder]


if __name__ == "__main__":
    args = argument_parser()
    print(args)
    labels = utils.load_labels(args.label_file)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    specical_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

    caption = " . ".join(labels) + " ."

    tokens, attention_mask, text_token_mask = text_tokenization(
        tokenizer, specical_tokens, caption
    )

    out = {}
    out["input_0"] = tokens[0]
    out["input_1"] = tokens[1]
    out["input_2"] = tokens[2]
    out["input_3"] = tokens[3]
    out["input_4"] = tokens[4]
    out["input_5"] = tokens[5]
    out["attention_mask"] = attention_mask
    out["text_token_mask"] = text_token_mask
    np.savez(args.output_file, **out)
    print(f"save result to: {args.output_file}")
