import argparse
import os
from typing import Dict, Iterable, List, Union
from threading import Thread, Event
from queue import Queue

import numpy as np

import vaststreamx as vsx


class NLPVastStreamX:
    def __init__(
        self,
        model_prefix_path: Union[str, Dict[str, str]],
        device_id: int = 0,
        batch_size: int = 1,
        is_async_infer: bool = False,
        model_output_op_name: str = ""
    ):

        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.input_id = 0

        self.attr = vsx.AttrKey
        assert vsx.set_device(self.device_id)==0
        # 构建model，模型三件套目录
        model_path = model_prefix_path
        self.model = vsx.Model(model_path, batch_size)
        # 输入预处理op
        self.embedding_op = vsx.Operator(vsx.OpType.BERT_EMBEDDING_OP)
        # 有以上op时无法载通过vsx.Operator加载vdsp算子
        # self.fusion_op = vsx.Operator.load_ops_from_json_file(vdsp_params_info)[0]

        # 构建graph
        self.graph = vsx.Graph()
        self.model_op = vsx.ModelOperator(self.model)
        self.graph.add_operators(self.embedding_op, self.model_op)

        # 构建stream
        self.infer_stream = vsx.Stream(self.graph, vsx.StreamBalanceMode.ONCE)
        if is_async_infer and len(self.model_output_op_name) > 0:
            self.infer_stream.register_operator_output(
                self.model_output_op_name, self.model_op)
        else:
            self.infer_stream.register_operator_output(self.model_op)

        # # 预处理算子输出
        self.infer_stream.build()

        self.current_id = -1
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}
        # 异步
        self.consumer = Thread(target=self.async_receive_infer)
        self.consumer.start()
        
    def async_receive_infer(self):
        while 1:
            try:
                result = None
                if len(self.model_output_op_name) > 0:
                    result = self.infer_stream.get_operator_output(
                        self.model_output_op_name)
                else:
                    result = self.infer_stream.get_operator_output(
                        self.model_op)
                if result is not None:
                    # pre_process_tensor = self.infer_stream.get_operator_output(self.preprocess_name)
                    # 输出顺序和输入一致
                    self.current_id += 1
                    input_id, = self.input_dict[self.current_id]
                    model_output_list = [
                        [vsx.as_numpy(out).astype(np.float32) for out in result[0]]]
                    self.result_dict[input_id] = []
                    self.post_processing(input_id, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                print(error_message)
                break
            
    def post_processing(self, input_id, stream_output_list):
        output_data = stream_output_list[0][0]

        self.result_dict[input_id].append(
            {
                "output": output_data,
            }
        )

    def get_datasets(self, npz_datalist_path: str, version=1.5):
        npz_datalist_fr = open(npz_datalist_path, 'r')
        npz_datalist = npz_datalist_fr.readlines()

        self.files_len = len(npz_datalist)
        if self.files_len == 0:
            raise ValueError('dataset files is None.')

        def dataset_loader():
            for index, data_path in enumerate(npz_datalist):
                inputs = np.load(data_path.strip())
                vsx_tensors = [
                    vsx.from_numpy(
                        np.array(input, dtype=np.int32), self.device_id) for _, input in inputs.items()
                ]
                ############## compiler 1.5+, 6个input################################
                if version > 1.3 and len(vsx_tensors) <= 3:
                    vsx_tensors.extend(
                        [
                            vsx.from_numpy(np.array(inputs[inputs.files[0]], dtype=np.int32), self.device_id)
                            for _ in range(6 - len(vsx_tensors))
                        ]
                    )
                yield vsx_tensors

        return dataset_loader

    def _run(self, vsx_tensors):
        input_id = self.input_id
        self.input_dict[input_id] = (input_id,)
        self.event_dict[input_id] = Event()
        
        self.infer_stream.run_async([vsx_tensors])
        self.input_id += 1
        return input_id
    
    def run_batch(self, datasets: Iterable[List[np.ndarray]]):
        queue = Queue(20)

        def input_thread():
            for data in datasets:
                input_id = self._run(data)
                queue.put(input_id)
            queue.put(None)

        thread = Thread(target=input_thread)
        thread.start()
        while True:
            input_id = queue.get()
            if input_id is None:
                break
            self.event_dict[input_id].wait()
            result = self.result_dict.pop(input_id)
            del self.event_dict[input_id]
            yield result
        
    def save(self, out, save_dir, name):
        outputs = {}
        outputs = {f'output_{i}': o['output'] for i, o in enumerate(out)}
        np.savez(os.path.join(save_dir, name), **outputs)
        
    def finish(self):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()
            
            
if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="RUN Det WITH VSX")
    parse.add_argument(
        "--data_list",
        type=str,
        default="./code/vmc/datasets/nlp/mrpc/npz_datalist.txt",
        help="img or dir path",
    )
    parse.add_argument(
        "--model_prefix_path",
        type=str,
        default="./code/vmc/deploy_weights/mrpc_ernie2_base_en_128-int8-max-1_128_1_128_1_128-vacc/mrpc_ernie2_base_en_128",
        help="model info"
    )
    parse.add_argument("--device_id", type=int, default=0, help="device id")
    parse.add_argument("--batch", type=int, default=4, help="bacth size")
    parse.add_argument("--save_dir", type=str,
                       default="./output", help="save_dir")
    args = parse.parse_args()

    sc = NLPVastStreamX(
        model_prefix_path=args.model_prefix_path,
        device_id=args.device_id,
        batch_size=args.batch,
        is_async_infer=False,
        model_output_op_name="",
    )
    datasets = sc.get_datasets(args.data_list)
    results = sc.run_batch(datasets())
    
    os.makedirs(args.save_dir, exist_ok=True)
    for i, result in enumerate(results):
        sc.save(result, args.save_dir, str(i).zfill(6))
        print(f"Num: {i}")
    
    sc.finish()
