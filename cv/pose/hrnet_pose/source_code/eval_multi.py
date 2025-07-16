
import os
import torch
import glob
import cv2
import argparse
import multiprocessing as mp
import numpy as np

from utils.utils import merge_configs
from data_process import data_process
from hrnet_postprocess import forward_test
from utils.bottom_up_coco import BottomUpCocoDataset
from utils.coco import dataset_info as coco_datainfo
from data_process import data_cfg, test_pipeline


def post_process(img, output, name2index, flip_test = True):
    # for img in img_l: ###
        
    img_name = os.path.basename(img).split('.')[0]
    imgdata= cv2.imread(img)
    h, w, _ = imgdata.shape
    pred_data = []
    output_flip = []

    index = name2index[img_name]
    npz_index = str(index).rjust(6,'0')
    toutput = np.load(os.path.join(output,'output_'+npz_index+'.npz'),allow_pickle=True)
    pred_data.append(torch.Tensor(toutput["output_0"]))

    npz_flip_index = name2index[img_name]+1
    npz_flip_index = str(npz_flip_index).rjust(6,'0')
    toutput_flip = np.load(os.path.join(output,'output_'+npz_flip_index+'.npz'),allow_pickle=True)
    output_flip.append(torch.Tensor(toutput_flip["output_0"]))

    # _ , img_metas , sigmas= data_process(img)
    img_metas= data_process(img)
    #print(img_metas)
    # exit()
    img_metas['base_size'] = (w, h)
    ncl = max(w,h)/200
    img_metas['scale'] = np.array([ncl, ncl])
    img_metas['center'] = (int(w/2),int(h/2))

    
    result = forward_test(outputs=pred_data,
        # img=img,
        outputs_flipped = output_flip,
        img_metas=img_metas,
        return_heatmap=False,
        flip_test=flip_test
    )
    # yield result  ###
    return result


def main(arg):
    img_dir = arg.data_root+"/val2017"
    imglist = glob.glob(img_dir+"/*")
    name2index = {}
    with open(args.datalist_txt, 'r') as f:
        for i, line in enumerate(f):
            name = os.path.basename(line.strip()).split('.')[0]
            name2index[name] = i


    print("Start feature postprocessing...")
    results = []
    tres = []

    from tqdm import tqdm
    pbar = tqdm(total=len(imglist))
    pbar.set_description('Sleep')
    update = lambda *args: pbar.update()

    n_proc = int(args.num_process)
    pool = mp.Pool(n_proc)
    for img in imglist:
        res = pool.apply_async(post_process, (img, args.output_data, name2index), callback=update)
        tres.append(res)
    for res in tres:
        results.append(res.get())
    pool.close()
    pool.join()
    # for img in imglist:
    #     res = post_process(img,args.output_data, name2index)


    eval_config = dict(interval=50, metric='mAP', save_best='AP')
    eval_config = merge_configs(eval_config, dict(metric='mAP'))
    
    dataset = BottomUpCocoDataset(
        ann_file=f'{args.data_root}/person_keypoints_val2017.json',
        img_prefix=f'{args.data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=coco_datainfo,
        test_mode=True
    )

    print("Accuracy calculation in progress")
    os.makedirs('./work_dir',mode=0o777, exist_ok=True)

    out_result = dataset.evaluate(results, './work_dir', **eval_config)
    
    for k, v in sorted(out_result.items()):
        print(f'{k}: {v}')

def parse_args():
    parser = argparse.ArgumentParser(description="Convert front model to vacc.")
    parser.add_argument('--flip_test',action="store_true" )
    parser.add_argument('--data_root',default= "./source_data/dataset/")
    parser.add_argument("--output_data",default= "./model_throughput_npz")
    parser.add_argument("--datalist_txt",default= "./data_npz_datalist.txt")
    parser.add_argument("--num_process",default=10)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)