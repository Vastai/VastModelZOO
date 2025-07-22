#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from typing import Union, List
from transformers import BertTokenizerFast

from model_nlp import ModelNLP, vsx


class UIEVacc(ModelNLP):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        tokenizer_path,
        max_seq_len=512,
        batch_size=1,
        device_id=0,
        hw_config="",
    ):
        super().__init__(
            model_prefix=model_prefix,
            vdsp_config=vdsp_config,
            batch_size=batch_size,
            device_id=device_id,
            hw_config=hw_config,
        )
        self.tokenizer_path_ = tokenizer_path
        # print(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer_ = BertTokenizerFast.from_pretrained(tokenizer_path)
        self.device_id_ = device_id
        self.max_seq_len_ = max_seq_len
        self.batch_size_ = batch_size

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        tokens = self.make_tokens('航母', '印媒所称的“印度第一艘国产航母”—“维克兰特”号')
        if ctx == "CPU":
            return [tokens] * batch_size
        else:
            return [
                [vsx.from_numpy(token, self.device_id_) for token in tokens]
            ] * batch_size

    def make_tokens(self, short_texts_prompts,short_input_texts):
        # print(f"prompt: {short_texts_prompts}, text: {short_input_texts}")
        if isinstance(short_texts_prompts, dict) or isinstance(short_texts_prompts, str):
            schema = [short_texts_prompts]
        # self._schema_tree = self._build_tree(schema)
        # schema_list = self._schema_tree.children[:]
        # print(f"type:{type(schema_list[0])}")
        if isinstance(short_input_texts, str):
            short_input_texts = [short_input_texts]

        # assert isinstance(short_input_texts, str), f"input type must be str"
        token_dict = self.tokenizer_(
            text=schema,
            text_pair=short_input_texts,
            stride=2,
            truncation=True,
            max_length=self.max_seq_len_,
            padding="longest",
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="np")
        

        token_list = token_dict["input_ids"][0]
        token_type_list = token_dict["token_type_ids"][0]
        attention_mask = token_dict["attention_mask"][0]
        # offset_maps = token_dict["offset_maps"][0]

        input_len = 512 # model_input_len ？
        # print(f"len:{len(token_list[0])}")
        seq_len = len(token_list)
        assert (
            seq_len <= input_len
        ), f"token len=({seq_len}) is larger than model max len=({input_len}),please input shorter string"
        
        input_ids = np.full([1, input_len], 0, dtype=np.int32)  #pad
        input_ids[0, :seq_len] = token_list

        token_type_ids = np.zeros(input_ids.shape, dtype=np.int32)
        token_type_ids[0, :seq_len] = token_type_list

        token_mask = np.zeros(input_ids.shape, dtype=np.int32)
        token_mask[0, :len(attention_mask)] = attention_mask
        zero_arr = np.zeros(input_ids.shape, dtype=np.int32)

        tokens = []
        tokens.append(input_ids)
        tokens.append(zero_arr)
        tokens.append(token_type_ids)
        tokens.append(token_mask)
        tokens.append(zero_arr)
        tokens.append(zero_arr)

        return tokens

    
    def process(
        self,
        input: Union[
            List[List[vsx.Tensor]],
            List[List[np.ndarray]],
            List[vsx.Tensor],
            List[np.ndarray],
            List[str],
            str,
        ],
    ):
        if isinstance(input, list):
            if isinstance(input[0], list):
                if isinstance(input[0][0], np.ndarray):
                    return self.process(
                        [
                            [
                                vsx.from_numpy(
                                    np.array(x, dtype=np.int32), self.device_id_
                                )
                                for x in one
                            ]
                            for one in input
                        ]
                    )
                else:
                    return self.process_impl(input)
            elif isinstance(input[0], str):
                return self.process([self.make_tokens(x) for x in input])
            elif isinstance(input[0], np.ndarray):
                tensors = [
                    vsx.from_numpy(np.array(x, dtype=np.int32), self.device_id_)
                    for x in input
                ]
                return self.process_impl([tensors])[0]
            else:
                return self.process_impl([input])[0]
        else:
            tokens = self.make_tokens(input)
            return self.process(tokens)

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        # return [[vsx.as_numpy(out).astype(np.float32) for out in output] for output in outputs]
        return [[vsx.as_numpy(out)for out in output] for output in outputs]

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


#     def process_test(self):
#         token_list = np.load('./input/input_ids.npy')[0]
#         token_type_list = np.load('./input/token_type_ids.npy')[0]
#         attention_mask = np.load('./input/attention_mask.npy')[0]
#         print(f"len:{len(token_list), len(attention_mask), len(token_type_list)}")
#         input_len = 512 # model_input_len ？
#         # print(f"len:{len(token_list[0])}")
#         seq_len = len(token_list)
#         assert (
#             seq_len <= input_len
#         ), f"token len=({seq_len}) is larger than model max len=({input_len}),please input shorter string"
        
#         input_ids = np.full([1, input_len], 0, dtype=np.int32)  #pad
#         input_ids[0, :seq_len] = token_list

#         token_type_ids = np.zeros(input_ids.shape, dtype=np.int32)
#         token_type_ids[0, :seq_len] = token_type_list

#         token_mask = np.zeros(input_ids.shape, dtype=np.int32)
#         token_mask[0, :len(attention_mask)] = attention_mask
#         zero_arr = np.zeros(input_ids.shape, dtype=np.int32)

#         tokens = []
#         tokens.append(input_ids)
#         tokens.append(zero_arr)
#         tokens.append(token_type_ids)
#         tokens.append(token_mask)
#         tokens.append(zero_arr)
#         tokens.append(zero_arr)

#         return self.process([tokens])
#         outputs = self.stream_.run_sync(input)
#         return [
#             [vsx.as_numpy(out).astype(np.float32) for out in output]
#             for output in outputs
#         ]


# ie = UIEVacc(
#     model_prefix="./build_model_2layer_work/gemma2b_iter0_2048_fp16",
#     vdsp_config="../../vdsp_config/bert_vdsp.json",
#     tokenizer_path="./config/uie_base_pytorch",
#     hw_config="")

# outputs = ie.process_test()
# start_prob = np.load('./input/start_prob.npy')
# end_prob = np.load('./input/end_prob.npy')
# print(f"len:{len(outputs)}, len:{len(outputs[0])}")
# exit(0)
# print(f"len:{len(outputs)}, shape:{outputs[0].shape}, shape2:{outputs[1].shape}")
# print(f"before start_prob:{start_prob.shape}, end_prob:{end_prob.shape}")
# vacc_start_prob = outputs[0][:start_prob.shape[-1], 0].reshape(1,-1)
# vacc_end_prob = outputs[1][:end_prob.shape[-1], 0].reshape(1,-1)
# np.save('start_prob_vsx.npy', vacc_start_prob)
# np.save('end_prob_vsx.npy', vacc_end_prob)
# print(f"vacc_start_prob:{vacc_start_prob.shape}, vacc_end_prob:{vacc_end_prob.shape}")
# print("cosine_sim start_prob:", cosine_similarity(vacc_start_prob, start_prob.reshape(1,-1)))
# print("cosine_sim end_prob:", cosine_similarity(vacc_end_prob, end_prob.reshape(1,-1)))
# print(f"before start_prob:{start_prob.reshape(1,-1).shape}, end_prob:{end_prob.reshape(1,-1).shape}")
# print(f"start_prob:{start_prob.reshape(1,-1)}, end_prob:{end_prob.reshape(1,-1)}")  
# print(f"tvm_output0:{vacc_start_prob}, tvm_output1:{vacc_end_prob}")

