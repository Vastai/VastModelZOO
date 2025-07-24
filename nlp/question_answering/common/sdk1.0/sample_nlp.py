
import argparse
import os

import numpy as np
from nlp_feature import NLP

parse = argparse.ArgumentParser(description="RUNSTREAM WITH VACL")
parse.add_argument(
    "--model_info", type=str, default='./code/model_profile/run_stream/model/squad/network.json', help="Required configuration file to run python runstream.")
parse.add_argument(
    "--bytes_size", type=int, default=1536, help="sequence byte length")
parse.add_argument(
    "--datalist_path", type=str, default='./code/model_profile/result/npz_datalist.txt', help="input *.npz datalist path")
parse.add_argument(
    "--device_id", type=int, default=0, help="device id")
parse.add_argument(
    "--batch_size", type=int, default=1, help="run batch num.")
parse.add_argument(
    "--save_dir", type=str, default='result/squad', help="save runstream results path")
args = parse.parse_args()


os.makedirs(args.save_dir, exist_ok=True)
 
# init model 
nlp = NLP(args.model_info, args.bytes_size, args.device_id, args.batch_size)

# build datasets iterator
datasets = nlp.get_datasets(args.datalist_path)

# batch run model, save result
results = nlp.run_batch(datasets())
for i, result in enumerate(results):
    out = {}
    out['output_0'] = result
    npz_id = str(i).zfill(6)
    np.savez(os.path.join(args.save_dir, 'output_' + npz_id), **out)
print()