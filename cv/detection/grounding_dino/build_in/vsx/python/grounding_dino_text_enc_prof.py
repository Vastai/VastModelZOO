
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

from groundingdino import GroundingDinoText
from groundingdino_prof import ModelProfiler
from easydict import EasyDict as edict
import argparse
import ast


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/path/to/text_encoder/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        help="hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--vdsp_params",
        default="/path/to/clip_txt_vdsp.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="/path/to/tokenizer/bert-base-uncased",
        help="tokenizer path",
    )
    parser.add_argument(
        "-d",
        "--device_ids",
        default="[0]",
        help="device ids to run",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=1,
        type=int,
        help="profiling batch size of the model",
    )
    parser.add_argument(
        "-i",
        "--instance",
        default=1,
        type=int,
        help="instance number for each device",
    )
    parser.add_argument(
        "--iterations",
        default=10240,
        type=int,
        help="iterations count for one profiling",
    )
    parser.add_argument(
        "--queue_size",
        default=1,
        type=int,
        help="aync wait queue size",
    )
    parser.add_argument(
        "--percentiles",
        default="[50, 90, 95, 99]",
        help="percentiles of latency",
    )
    parser.add_argument(
        "--input_host",
        default=0,
        type=int,
        help="cache input data into host memory",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    model_prefix = args.model_prefix
    vdsp_params = args.vdsp_params
    hw_config = args.hw_config
    tokenizer = args.tokenizer_path
    device_ids = ast.literal_eval(args.device_ids)
    batch_size = args.batch_size
    instance = args.instance
    iterations = args.iterations
    queue_size = args.queue_size
    input_host = args.input_host
    percentiles = ast.literal_eval(args.percentiles)

    models = []
    contexts = []

    for i in range(instance):
        device_id = device_ids[i % len(device_ids)]
        model = GroundingDinoText(
            model_prefix, vdsp_params, tokenizer, batch_size, device_id, hw_config
        )
        models.append(model)
        if input_host:
            contexts.append("CPU")
        else:
            contexts.append("VACC")

    # shape =
    shape = model.input_shape[:6]
    config = edict(
        {
            "instance": instance,
            "iterations": iterations,
            "batch_size": batch_size,
            "data_type": "int32",
            "device_ids": device_ids,
            "contexts": contexts,
            "input_shape": shape,
            "percentiles": percentiles,
            "queue_size": queue_size,
        }
    )
    profiler = ModelProfiler(config, models)
    print(profiler.profiling())
