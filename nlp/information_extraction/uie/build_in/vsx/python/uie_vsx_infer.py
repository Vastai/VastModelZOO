#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from model_nlp import ModelNLP, vsx

import re
import numpy as np
from typing import Union, List
# from transformers import AutoTokenizer
from transformers import BertTokenizerFast
from utils_uie import logger, get_bool_ids_greater_than, get_span, get_id_and_prob, cut_chinese_sent, dbc2sbc
from uie_vacc import UIEVacc


class UIEPredictorVacc(object):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        schema,
        tokenizer_path,
        schema_lang="zh",
        position_prob=0.5,
        max_seq_len=512,
        batch_size=4,
        split_sentence=False,
        device_id=0,
        hw_config="",
    ):
        self._tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        self._tokenizer_path = tokenizer_path
        self._device_id = device_id
        self._position_prob = position_prob
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._split_sentence = split_sentence

        self._schema_tree = None
        self._uie_vacc = UIEVacc(
            model_prefix=model_prefix,
            vdsp_config=vdsp_config,
            tokenizer_path=self._tokenizer_path,
            hw_config=hw_config,
            max_seq_len = max_seq_len,
            device_id=device_id)
        self._is_en = True if schema_lang == 'en' else False
        self.set_schema(schema)
    def set_schema(self, schema):
        if isinstance(schema, dict) or isinstance(schema, str):
            schema = [schema]
        self._schema_tree = self._build_tree(schema)

    def __call__(self, inputs):
        texts = inputs
        if isinstance(texts, str):
            texts = [texts]
        results = self._multi_stage_predict(texts)
        return results
    
    def _multi_stage_predict(self, datas):
        """
        Traversal the schema tree and do multi-stage prediction.
        Args:
            datas (list): a list of strings
        Returns:
            list: a list of predictions, where the list's length
                equals to the length of `datas`
        """
        results = [{} for _ in range(len(datas))]
        # input check to early return
        if len(datas) < 1 or self._schema_tree is None:
            return results

        # copy to stay `self._schema_tree` unchanged
        schema_list = self._schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0
            if not node.prefix:
                for data in datas:
                    examples.append({
                        "text": data,
                        "prompt": dbc2sbc(node.name)
                    })
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                for pre, data in zip(node.prefix, datas):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            if self._is_en:
                                if re.search(r'\[.*?\]$', node.name):
                                    prompt_prefix = node.name[:node.name.find(
                                        "[", 1)].strip()
                                    cls_options = re.search(
                                        r'\[.*?\]$', node.name).group()
                                    # Sentiment classification of xxx [positive, negative]
                                    prompt = prompt_prefix + p + " " + cls_options
                                else:
                                    prompt = node.name + p
                            else:
                                prompt = p + node.name
                            examples.append({
                                "text": data,
                                "prompt": dbc2sbc(prompt)
                            })
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1
            if len(examples) == 0:
                result_list = []
            else:
                result_list = self._single_stage_predict(examples)

            if not node.parent_relations:
                relations = [[] for i in range(len(datas))]
                for k, v in input_map.items():
                    for idx in v:
                        if len(result_list[idx]) == 0:
                            continue
                        if node.name not in results[k].keys():
                            results[k][node.name] = result_list[idx]
                        else:
                            results[k][node.name].extend(result_list[idx])
                    if node.name in results[k].keys():
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if "relations" not in relations[k][i].keys():
                            relations[k][i]["relations"] = {
                                node.name: result_list[v[i]]
                            }
                        elif node.name not in relations[k][i]["relations"].keys(
                        ):
                            relations[k][i]["relations"][
                                node.name] = result_list[v[i]]
                        else:
                            relations[k][i]["relations"][node.name].extend(
                                result_list[v[i]])

                new_relations = [[] for i in range(len(datas))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if "relations" in relations[i][j].keys(
                        ) and node.name in relations[i][j]["relations"].keys():
                            for k in range(
                                    len(relations[i][j]["relations"][
                                        node.name])):
                                new_relations[i].append(relations[i][j][
                                    "relations"][node.name][k])
                relations = new_relations

            prefix = [[] for _ in range(len(datas))]
            for k, v in input_map.items():
                for idx in v:
                    for i in range(len(result_list[idx])):
                        if self._is_en:
                            prefix[k].append(" of " +
                                             result_list[idx][i]["text"])
                        else:
                            prefix[k].append(result_list[idx][i]["text"] + "的")

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)
        return results

    def _convert_ids_to_results(self, examples, sentence_ids, probs):
            """
            Convert ids to raw text in a single stage.
            """
            results = []
            for example, sentence_id, prob in zip(examples, sentence_ids, probs):
                if len(sentence_id) == 0:
                    results.append([])
                    continue
                result_list = []
                text = example["text"]
                prompt = example["prompt"]
                for i in range(len(sentence_id)):
                    start, end = sentence_id[i]
                    if start < 0 and end >= 0:
                        continue
                    if end < 0:
                        start += (len(prompt) + 1)
                        end += (len(prompt) + 1)
                        result = {"text": prompt[start:end],
                                "probability": prob[i]}
                        result_list.append(result)
                    else:
                        result = {
                            "text": text[start:end],
                            "start": start,
                            "end": end,
                            "probability": prob[i]
                        }
                        result_list.append(result)
                results.append(result_list)
            return results

    def _auto_splitter(self, input_texts, max_text_len, split_sentence=False):
        '''
        Split the raw texts automatically for model inference.
        Args:
            input_texts (List[str]): input raw texts.
            max_text_len (int): cutting length.
            split_sentence (bool): If True, sentence-level split will be performed.
        return:
            short_input_texts (List[str]): the short input texts for model inference.
            input_mapping (dict): mapping between raw text and short input texts.
        '''
        input_mapping = {}
        short_input_texts = []
        cnt_org = 0
        cnt_short = 0
        for text in input_texts:
            if not split_sentence:
                sens = [text]
            else:
                sens = cut_chinese_sent(text)
            for sen in sens:
                lens = len(sen)
                if lens <= max_text_len:
                    short_input_texts.append(sen)
                    if cnt_org not in input_mapping.keys():
                        input_mapping[cnt_org] = [cnt_short]
                    else:
                        input_mapping[cnt_org].append(cnt_short)
                    cnt_short += 1
                else:
                    temp_text_list = [
                        sen[i:i + max_text_len]
                        for i in range(0, lens, max_text_len)
                    ]
                    short_input_texts.extend(temp_text_list)
                    short_idx = cnt_short
                    cnt_short += math.ceil(lens / max_text_len)
                    temp_text_id = [
                        short_idx + i for i in range(cnt_short - short_idx)
                    ]
                    if cnt_org not in input_mapping.keys():
                        input_mapping[cnt_org] = temp_text_id
                    else:
                        input_mapping[cnt_org].extend(temp_text_id)
            cnt_org += 1
        return short_input_texts, input_mapping

    def _single_stage_predict(self, inputs):
        input_texts = []
        prompts = []
        for i in range(len(inputs)):
            input_texts.append(inputs[i]["text"])
            prompts.append(inputs[i]["prompt"])
        # max predict length should exclude the length of prompt and summary tokens
        max_predict_len = self._max_seq_len - len(max(prompts)) - 3

        short_input_texts, self.input_mapping = self._auto_splitter(
            input_texts, max_predict_len, split_sentence=self._split_sentence)

        short_texts_prompts = []
        for k, v in self.input_mapping.items():
            short_texts_prompts.extend([prompts[k] for i in range(len(v))])
        short_inputs = [{
            "text": short_input_texts[i],
            "prompt": short_texts_prompts[i]
        } for i in range(len(short_input_texts))]

        sentence_ids = []
        probs = []

        input_ids = []
        token_type_ids = []
        attention_mask = []
        offset_maps = []

        padding_type = "longest"
        # print(f"short_texts_prompts:{short_texts_prompts},short_input_texts:{short_input_texts} ")
        encoded_inputs = self._tokenizer(
            text=short_texts_prompts,
            text_pair=short_input_texts,
            stride=2,
            truncation=True,
            max_length=self._max_seq_len,
            padding=padding_type,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="np")

        start_prob_concat, end_prob_concat = [], []
        for batch_start in range(0, len(short_input_texts), self._batch_size):
            input_ids = encoded_inputs["input_ids"][batch_start:batch_start+self._batch_size]
            # print(f"lens:{len(input_ids)},input_ids: {input_ids}")
            token_type_ids = encoded_inputs["token_type_ids"][batch_start:batch_start+self._batch_size]
            # print(f"lens:{len(token_type_ids)},token_type_ids: {token_type_ids}")
            attention_mask = encoded_inputs["attention_mask"][batch_start:batch_start+self._batch_size]
            # print(f"lens:{len(attention_mask)},attention_mask: {attention_mask}")
            offset_maps = encoded_inputs["offset_mapping"][batch_start:batch_start+self._batch_size]

            # input_dict = {
            #     "input_ids": np.array(
            #         input_ids, dtype="int64"),
            #     "token_type_ids": np.array(
            #         token_type_ids, dtype="int64"),
            #     "attention_mask": np.array(
            #         attention_mask, dtype="int64")
            # }
            model_len = 512 # model_input_len ？
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
            outputs = self._uie_vacc.process(multi_tokens)
            for i in range(input_id_batch):
                seq_len = len(input_ids[i])
                output = outputs[i]
                start_prob, end_prob = output[0][:seq_len, 0].reshape(1,-1), output[1][:seq_len, 0].reshape(1,-1)
                start_prob_concat.append(start_prob)
                end_prob_concat.append(end_prob)
        start_prob_concat = np.concatenate(start_prob_concat)
        end_prob_concat = np.concatenate(end_prob_concat)

        start_ids_list = get_bool_ids_greater_than(
            start_prob_concat, limit=self._position_prob, return_prob=True)
        end_ids_list = get_bool_ids_greater_than(
            end_prob_concat, limit=self._position_prob, return_prob=True)

        # input_ids = input_dict['input_ids']
        sentence_ids = []
        probs = []
        for start_ids, end_ids, ids, offset_map in zip(start_ids_list,
                                                       end_ids_list,
                                                       input_ids.tolist(),
                                                       offset_maps):
            for i in reversed(range(len(ids))):
                if ids[i] != 0:
                    ids = ids[:i]
                    break
            span_list = get_span(start_ids, end_ids, with_prob=True)
            sentence_id, prob = get_id_and_prob(span_list, offset_map.tolist())
            sentence_ids.append(sentence_id)
            probs.append(prob)

        results = self._convert_ids_to_results(short_inputs, sentence_ids,
                                               probs)
        results = self._auto_joiner(results, short_input_texts,
                                    self.input_mapping)
        return results

    
    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        is_cls_task = False
        for short_result in short_results:
            if short_result == []:
                continue
            elif 'start' not in short_result[0].keys(
            ) and 'end' not in short_result[0].keys():
                is_cls_task = True
                break
            else:
                break
        for k, vs in input_mapping.items():
            if is_cls_task:
                cls_options = {}
                single_results = []
                for v in vs:
                    if len(short_results[v]) == 0:
                        continue
                    if short_results[v][0]['text'] not in cls_options.keys():
                        cls_options[short_results[v][0][
                            'text']] = [1, short_results[v][0]['probability']]
                    else:
                        cls_options[short_results[v][0]['text']][0] += 1
                        cls_options[short_results[v][0]['text']][
                            1] += short_results[v][0]['probability']
                if len(cls_options) != 0:
                    cls_res, cls_info = max(cls_options.items(),
                                            key=lambda x: x[1])
                    concat_results.append([{
                        'text': cls_res,
                        'probability': cls_info[1] / cls_info[0]
                    }])
                else:
                    concat_results.append([])
            else:
                offset = 0
                single_results = []
                for v in vs:
                    if v == 0:
                        single_results = short_results[v]
                        offset += len(short_inputs[v])
                    else:
                        for i in range(len(short_results[v])):
                            if 'start' not in short_results[v][
                                    i] or 'end' not in short_results[v][i]:
                                continue
                            short_results[v][i]['start'] += offset
                            short_results[v][i]['end'] += offset
                        offset += len(short_inputs[v])
                        single_results.extend(short_results[v])
                concat_results.append(single_results)
        return concat_results

    # def predict(self, input_data):
    #     results = self._multi_stage_predict(input_data)
    #     return results

    # def get_test_data(self, dtype, input_shape, _batch_size, ctx="VACC"):
    #     tokens = self.make_tokens("test string")
    #     if ctx == "CPU":
    #         return [tokens] * _batch_size
    #     else:
    #         return [
    #             [vsx.from_numpy(token, self._device_id) for token in tokens]
    #         ] * _batch_size
    # def make_tokens(self, text):
    #     assert isinstance(text, str), f"input type must be str"
    #     # token_dict = self._tokenizer(text=text, return_tensors="pt", padding=True)
    #     encoded_inputs = self._tokenizer(
    #         text=short_texts_prompts,
    #         text_pairs=text,
    #         stride=2,
    #         truncation=True,
    #         max_length=self._max_seq_len,
    #         padding="longest",
    #         add_special_tokens=True,
    #         return_offsets_mapping=True,
    #         return_tensors="np")
    #     for batch_start in range(0, len(short_input_texts), self._batch_size):
    #         input_ids = encoded_inputs["input_ids"][batch_start:batch_start+self._batch_sie]
    #         token = token_dict["input_ids"][0]
    #         input_seq_len = 16
    #         token_padding = np.full([input_seq_len], 49407, dtype=np.int32)  # pad
    #         token_padding[: len(token)] = token
    #         # make mask
    #         token_mask = np.ones(shape=(input_seq_len), dtype=np.int32) * (-1)
    #         mask = token_dict["attention_mask"][0]
    #         token_mask[: len(mask)] = mask
    #         # make input
    #         zero_arr = np.zeros(token_padding.shape, dtype=np.int32)
    #         tokens = []
    #         tokens.append(token_padding)
    #         tokens.append(zero_arr)
    #         tokens.append(zero_arr)
    #         tokens.append(token_mask)
    #         tokens.append(zero_arr)
    #         tokens.append(zero_arr)

    #     return tokens

    # def process(
    #     self,
    #     input: Union[
    #         List[List[vsx.Tensor]],
    #         List[List[np.ndarray]],
    #         List[vsx.Tensor],
    #         List[np.ndarray],
    #         List[str],
    #         str,
    #     ],
    # ):
    #     if isinstance(input, list):
    #         if isinstance(input[0], list):
    #             if isinstance(input[0][0], np.ndarray):
    #                 return self.process(
    #                     [
    #                         [
    #                             vsx.from_numpy(
    #                                 np.array(x, dtype=np.int32), self._device_id
    #                             )
    #                             for x in one
    #                         ]
    #                         for one in input
    #                     ]
    #                 )
    #             else:
    #                 return self.process_impl(input)
    #         elif isinstance(input[0], str):
    #             return self.process([self.make_tokens(x) for x in input])
    #         elif isinstance(input[0], np.ndarray):
    #             tensors = [
    #                 vsx.from_numpy(np.array(x, dtype=np.int32), self._device_id)
    #                 for x in input
    #             ]
    #             return self.process_impl([tensors])[0]
    #         else:
    #             return self.process_impl([input])[0]
    #     else:
    #         tokens = self.make_tokens(input)
    #         return self.process(tokens)

    # def process_impl(self, input):
    #     outputs = self.stream_.run_sync(input)
    #     return [
    #         [vsx.as_numpy(out).astype(np.float32) for out in output]
    #         for output in outputs
    #     ]

    @classmethod
    def _build_tree(cls, schema, name='root'):
        """
        Build the schema tree.
        """
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, str):
                        child = [v]
                    elif isinstance(v, list):
                        child = v
                    else:
                        raise TypeError(
                            "Invalid schema, value for each key:value pairs should be list or string"
                            "but {} received".format(type(v)))
                    schema_tree.add_child(cls._build_tree(child, name=k))
            else:
                raise TypeError(
                    "Invalid schema, element should be string or dict, "
                    "but {} received".format(type(s)))
        return schema_tree
    
class SchemaTree(object):
    """
    Implementataion of SchemaTree
    """

    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        self.prefix = None
        self.parent_relations = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(
            node, SchemaTree
        ), "The children of a node should be an instacne of SchemaTree."
        self.children.append(node)

import argparse
def argument_parser():
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
        "--schema",
        default="'航母'",
        help="schema (default: %(default)s)",
    )
    parser.add_argument(
        "--input_txt",
        default="'印媒所称的“印度第一艘国产航母”—“维克兰特”号'",
        help="inputs str (default: %(default)s)",
    )
    parser.add_argument(
        "--schema_lang",
        default="ch",
        help="schema_lang, ch or en (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size (default: %(default)s)",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    assert vsx.set_device(args.device_id) == 0
    schema = [args.schema]
    schema_lang = args.schema_lang
    model_prefix = args.model_prefix
    vdsp_config = args.vdsp_params
    hw_config = args.hw_config
    batch_size = args.batch_size
    tokenizer_path = args.tokenizer_path

    uie = UIEPredictorVacc(
        model_prefix=model_prefix, 
        vdsp_config=vdsp_config,
        tokenizer_path=tokenizer_path,
        hw_config=hw_config,
        batch_size=batch_size,
        schema_lang=schema_lang, 
        schema=schema,
        device_id=args.device_id)
    print(uie(args.input_txt))
