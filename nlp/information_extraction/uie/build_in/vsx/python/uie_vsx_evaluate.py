# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from uie_vacc import UIEVacc
import argparse
from functools import partial
import numpy as np

import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from utils_uie import IEMapDataset, SpanEvaluator, IEDataset, convert_example, get_relation_type_dict, logger, tqdm, unify_prompt_name


@torch.no_grad()
def evaluate(model, metric, data_loader, loss_fn=None, show_bar=True):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`torch.nn.Module`): A model to classify texts.
        metric(obj:`Metric`): The evaluation metric.
        data_loader(obj:`torch.utils.data.DataLoader`): The dataset loader which generates batches.
    """
    return_loss = False
    if loss_fn is not None:
        return_loss = True
    # model.eval()
    metric.reset()
    loss_list = []
    loss_sum = 0
    loss_num = 0
    if show_bar:
        data_loader = tqdm(
            data_loader, desc="Evaluating", unit='batch')
    for batch in data_loader:
        input_ids, token_type_ids, att_mask, start_ids, end_ids = batch
        # if device == 'gpu':
        #     input_ids = input_ids.cuda()
        #     token_type_ids = token_type_ids.cuda()
        #     att_mask = att_mask.cuda()

        model_len = 512 # model_input_len ？
        input_ids = input_ids.numpy()
        token_type_ids = token_type_ids.numpy()
        attention_mask = att_mask.numpy()
        input_id_batch = len(input_ids)
        multi_tokens = []
        for i in range(input_id_batch):
            seq_len = len(input_ids[i])
            assert (seq_len <= model_len
            ), f"token len=({seq_len}) is larger than model max len=({model_len}),please input shorter string"
            token_input_ids = np.full([1, model_len], 0, dtype=np.int32)  #pad
            token_input_ids[:, :seq_len] = input_ids[i]

            type_ids = np.zeros(token_input_ids.shape, dtype=np.int32)
            type_ids[:, :seq_len] = token_type_ids[i]   

            token_mask = np.ones(token_input_ids.shape, dtype=np.int32)
            token_mask[:, :seq_len] = attention_mask[i]

            zero_arr = np.zeros(token_input_ids.shape, dtype=np.int32)

            tokens = []
            tokens.append(token_input_ids)
            tokens.append(zero_arr)
            tokens.append(type_ids)
            tokens.append(token_mask)
            tokens.append(zero_arr)
            tokens.append(zero_arr)
            multi_tokens.append(tokens)

        # outputs = self.inference_backend.infer(input_dict)
        outputs = model.process(multi_tokens)
        # print(f"len:{len(outputs)}, input_len:{input_id_batch}")
        # exit(1)
        start_prob_concat, end_prob_concat = [], []
        for i in range(input_id_batch):
            seq_len = len(input_ids[i])
            output = outputs[i]
            start_prob, end_prob = torch.tensor(output[0][:seq_len, 0].reshape(1,-1)), torch.tensor(output[1][:seq_len, 0].reshape(1,-1))
            
            # print(f"i:{i}, start_prob:{start_prob}, end_prob:{end_prob}")
            start_prob_concat.append(start_prob)
            end_prob_concat.append(end_prob)
        
        # outputs = model(input_ids=input_ids,
        #                 token_type_ids=token_type_ids,
        #                 attention_mask=att_mask)
        # start_prob, end_prob = outputs[0], outputs[1]

        # if device == 'gpu':
        #     start_prob, end_prob = start_prob.cpu(), end_prob.cpu()
        start_ids = start_ids.type(torch.float32)
        end_ids = end_ids.type(torch.float32)
        start_prob_concat = np.concatenate(start_prob_concat)
        end_prob_concat = np.concatenate(end_prob_concat)
        if return_loss:
            # Calculate loss
            loss_start = loss_fn(start_prob_concat, start_ids)
            loss_end = loss_fn(end_prob_concat, end_ids)
            loss = (loss_start + loss_end) / 2.0
            loss = float(loss)
            loss_list.append(loss)
            loss_sum += loss
            loss_num += 1
            if show_bar:
                data_loader.set_postfix(
                    {
                        'dev loss': f'{loss_sum / loss_num:.5f}'
                    }
                )

        # Calcalate metric
        num_correct, num_infer, num_label = metric.compute(start_prob_concat, end_prob_concat,
                                                           start_ids, end_ids)
        metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    # model.train()
    if return_loss:
        loss_avg = sum(loss_list) / len(loss_list)
        return loss_avg, precision, recall, f1
    else:
        return precision, recall, f1


def do_eval():
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)
    # model = UIE.from_pretrained(args.model_path)
    model = UIEVacc(
        model_prefix=args.model_prefix,
        vdsp_config=args.vdsp_params,
        tokenizer_path=args.tokenizer_path,
        hw_config=args.hw_config,
        device_id=args.device_id,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len)
    # if args.device == 'gpu':
    #     model = model.cuda()

    test_ds = IEDataset(args.test_path, tokenizer=tokenizer,
                        max_seq_len=args.max_seq_len)

    test_data_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False)
    class_dict = {}
    relation_data = []
    if args.debug:
        for data in test_ds.dataset:
            class_name = unify_prompt_name(data['prompt'])
            # Only positive examples are evaluated in debug mode
            if len(data['result_list']) != 0:
                if "的" not in data['prompt']:
                    class_dict.setdefault(class_name, []).append(data)
                else:
                    relation_data.append((data['prompt'], data))
        relation_type_dict = get_relation_type_dict(relation_data)
    else:
        class_dict["all_classes"] = test_ds

    for key in class_dict.keys():
        if args.debug:
            test_ds = IEMapDataset(class_dict[key], tokenizer=tokenizer,
                                   max_seq_len=args.max_seq_len)
        else:
            test_ds = class_dict[key]

        test_data_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False)
        metric = SpanEvaluator()
        precision, recall, f1 = evaluate(
            model, metric, test_data_loader)
        logger.info("-----------------------------")
        logger.info("Class Name: %s" % key)
        logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" %
                    (precision, recall, f1))

    if args.debug and len(relation_type_dict.keys()) != 0:
        for key in relation_type_dict.keys():
            test_ds = IEMapDataset(relation_type_dict[key], tokenizer=tokenizer,
                                   max_seq_len=args.max_seq_len)

            test_data_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False)
            metric = SpanEvaluator()
            precision, recall, f1 = evaluate(
                model, metric, test_data_loader)
            logger.info("-----------------------------")
            logger.info("Class Name: X的%s" % key)
            logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" %
                        (precision, recall, f1))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_prefix",
        default="./build_model_2layer_work/gemma2b_iter0_2048_fp16",
        help="model prefix of the model suite files (default: %(default)s)",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite (default: %(default)s)",
    )
    parser.add_argument(
        "--vdsp_params",
        default="../../vdsp_config/bert_vdsp.json",
        help="vdsp preprocess parameter file (default: %(default)s)",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="./config/uie_base_pytorch",
        help="tokenizer path (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size (default: %(default)s)",
    )

    parser.add_argument("-t", "--test_path", type=str, default="./data/dev.txt",
                        help="The path of test set. (default: %(default)s)")

    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after tokenization. (default: %(default)s)")
    
    parser.add_argument("--debug", action='store_true',
                        help="Precision, recall and F1 score are calculated for each class separately if this option is enabled.")

    args = parser.parse_args()
    # yapf: enable

    do_eval()
