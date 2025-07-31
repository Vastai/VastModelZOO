# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import argparse
import os
from queue import Queue
from threading import Event, Thread
from typing import Dict, Iterable, List, Union

import numpy as np
import torch


class Bert:
    def __init__(self, model_prefix_path: str):
        self.model = torch.jit.load(model_prefix_path)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.model.eval()
        self.model.to(self.device_id)

    def run(self, input_data: List[torch.Tensor]):
        # for i in range(len(input_data)):
        #     print("dtype", input_data[i].dtype)
        #     print("shape", input_data[i].shape)
        with torch.no_grad():
            # combined_tensor = torch.stack(input_data, dim=0)
            # heatmap = self.model(combined_tensor)
            input_data = [
                torch.unsqueeze(tensor, 0).to(self.device_id) for tensor in input_data
            ]
            output = self.model(input_data[0], input_data[1], input_data[2])
            print("finish")
            # print("output", output)
        return [{"output": output[0].cpu().numpy()}]

    def run_batch(self, datasets: Iterable[List[torch.Tensor]]):
        for data in datasets:
            yield self.run(data)

    def get_datasets(self, npz_datalist_path: str, version=1.5):
        npz_datalist_fr = open(npz_datalist_path, "r")
        npz_datalist = npz_datalist_fr.readlines()

        self.files_len = len(npz_datalist)
        if self.files_len == 0:
            raise ValueError("dataset files is None.")

        def dataset_loader():
            for index, data_path in enumerate(npz_datalist):
                inputs = np.load(data_path.strip())
                print(f"load {data_path.strip()}")
                # for _, input in inputs.items():
                #     print("dtype", input.dtype)
                #     print("shape", input.shape)
                torch_tensors = [
                    torch.from_numpy(np.array(input, dtype=np.int32))
                    for _, input in inputs.items()
                ]
                ############## compiler 1.5+, 6个input################################
                if version > 1.3 and len(torch_tensors) <= 3:
                    torch_tensors.extend(
                        [
                            torch.from_numpy(
                                np.array(inputs[inputs.files[0]], dtype=np.int32)
                            )
                            for _ in range(6 - len(torch_tensors))
                        ]
                    )
                yield torch_tensors

        return dataset_loader

    def save(self, out, save_dir, name):
        outputs = {}
        outputs = {f"output_{i}": o["output"] for i, o in enumerate(out)}
        np.savez(os.path.join(save_dir, name), **outputs)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="RUN Det WITH VSX")
    parse.add_argument(
        "--data_list",
        type=str,
        default="/path/to/npz_datalist.txt",
        help="img or dir path",
    )
    parse.add_argument(
        "--model_path",
        type=str,
        default="/path/to/model.torchscript.pt",
        help="model info",
    )
    parse.add_argument("--save_dir", type=str, default="./output", help="save_dir")
    args = parse.parse_args()

    # sc = NLPVastStreamX(
    #     model_prefix_path=args.model_prefix_path,
    #     device_id=args.device_id,
    #     batch_size=args.batch,
    #     is_async_infer=False,
    #     model_output_op_name="",
    # )
    sc = Bert(args.model_path)
    datasets = sc.get_datasets(args.data_list)
    results = sc.run_batch(datasets())

    os.makedirs(args.save_dir, exist_ok=True)
    for i, result in enumerate(results):
        sc.save(result, args.save_dir, str(i).zfill(6))
        print(f"Num: {i}")
