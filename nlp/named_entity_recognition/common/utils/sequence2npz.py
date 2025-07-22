import os
from typing import Dict
import argparse

import numpy as np


def val_npz_result(npz_path):
    npz_data = np.load(npz_path)
    print(npz_data.files)
    print(npz_data[npz_data.files[0]])
    print(npz_data[npz_data.files[1]])
    print(npz_data[npz_data.files[2]])
    
    print(npz_data[npz_data.files[0]].shape)
    print(npz_data[npz_data.files[1]].shape)
    print(npz_data[npz_data.files[2]].shape)
    

def run_bin2npz(bin_root, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(os.path.dirname(save_dir), 'npz_datalist.txt')
    bin_dirs = os.listdir(bin_root)
    bin_dirs.sort()
    with open(txt_path, 'w') as fw:
        for i in range(len(bin_dirs)):
            dir_name = 'test__' + str(i)
            save_path = save_dir + '/test_' + str(i) + '.npz'
            bin2npz_val(bin_root + '/' + dir_name, save_path)
            print(dir_name)
            fw.write(save_path + '\n')
        
    
def bin2npz_val(data_dir, npz_path):
    features = {}
    input_ids = np.fromfile(os.path.join(data_dir, 'input_ids_1.bin'), dtype=np.int32)
    token_type_ids = np.fromfile(os.path.join(data_dir, 'segment_ids_2.bin'), dtype=np.int32)
    attention_mask = np.fromfile(os.path.join(data_dir, 'input_mask_1.bin'), dtype=np.int32)
    
    features['input_0'] = input_ids
    features['input_1'] = token_type_ids
    features['input_2'] = attention_mask
    
    np.savez(npz_path, **features)
    

def wrtie_datalist(npz_path, save_dir):
    npz_files = os.listdir(npz_path)
    files_len = len(npz_files)
    with open(save_dir, 'w') as fw:
        for i in range(files_len):
            path= os.path.join(npz_path, 'test_' + str(i) + '.npz')
            fw.write(path + '\n')
            print(path)
        


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="DATA FORMAT CONVERT")
    parse.add_argument(
        "--npz_path",
        type=str,
        default="/home/jies/code/modelzoo/model_test/datasets/dev408_3inputs",
        help="MRPC-dev *.bin data path",
    )
    parse.add_argument(
        "--save_path", 
        type=str,
        default="/home/jies/code/modelzoo/model_test/datasets/dev408_3inputs.txt",
        help="output *.npz and npz_datalist.txt path"
    )
    args = parse.parse_args()
    
    # run_bin2npz(args.bin_path, args.save_path)
    
    wrtie_datalist(args.npz_path, args.save_path)
    
    









