import sys
import os
import torch
import numpy as np

_cur_file_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(_cur_file_path + os.sep + '../../source_code/fast-reid/')

from fastreid.config import get_cfg
from collections import OrderedDict
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.DEVICE = "cpu"

    with open("data_list.txt") as f:
        datalist = f.readlines()

    datalist = [x.strip() for x in datalist]

    output_npz_list = sorted(os.listdir("npz_out/"))

    results = OrderedDict()
    for _, dataset_name in enumerate(cfg.DATASETS.TESTS):
        data_loader, evaluator = DefaultTrainer.build_evaluator(cfg, dataset_name)
        
        for idx, inputs in enumerate(data_loader):

            outputs = []
            for i, input_data in enumerate(inputs['img_paths']):

                # print(input_data.split('/')[-1].split('.')[0])
                # # print(datalist[i].split('/')[-1].split('.')[0])
                # print(output_npz_list[i+idx*128])

                assert input_data.split('/')[-1].split('.')[0] == datalist[i+idx*128].split('/')[-1].split('.')[0]

                output = np.load(os.path.join("npz_out/", output_npz_list[i+idx*128]), allow_pickle=True)["output_0"]
                outputs.append(output)

            outputs = torch.Tensor(outputs).squeeze(1)
            evaluator.process(inputs, outputs)

        results[dataset_name] = evaluator.evaluate()

    if len(results) == 1:
        results = list(results.values())[0]
    
    print(results)
            
    return results

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )