import os
import cv2
import glob
import torch
import hashlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F

import tvm
import vacc
from tvm.contrib import graph_runtime


def get_activation_aligned_faster(activation, dtype=np.float16, fc_mode=False, force_int8_layout_to_fp16=False):  # NCHW
    N = C = H = W = 1
    if len(activation.shape) == 2:
        N, C = activation.shape
        fc_mode = True
    elif len(activation.shape) == 5:
        N, C, H, W, B = activation.shape
    elif len(activation.shape) == 1:
        (C,) = activation.shape
    else:
        N, C, H, W = activation.shape
    h_group = w_group = c_group = 0
    if H == 1 and W == 1 and fc_mode == True:
        if dtype == np.float16:
            h_group, w_group, c_group = 1, 1, 256
        elif dtype == np.int8:
            h_group, w_group, c_group = 1, 1, 512
    else:
        if dtype == np.float16 or force_int8_layout_to_fp16:
            h_group, w_group, c_group = 8, 8, 4
        elif dtype == np.int8:
            h_group, w_group, c_group = 8, 8, 8
    pad_H, pad_W, pad_C = H, W, C
    pad_h, pad_w = 0, 0
    if H % h_group != 0:
        pad_h = h_group - H % h_group
        pad_H += pad_h
    if W % w_group != 0:
        pad_w = w_group - W % w_group
        pad_W += pad_w
    if C % c_group != 0:
        pad_c = c_group - C % c_group
        pad_C += pad_c
    # tensorize to WHC4c8h8w
    w_num = pad_W // w_group
    h_num = pad_H // h_group
    c_num = pad_C // c_group
    n_num = N
    block_size = w_group * h_group * c_group
    activation = activation.astype(dtype)
    assert(len(activation.shape) == 4)
    if (pad_h | pad_w) != 0:
        activation = np.pad(activation, ((0,0),(0,0),(0,pad_h),(0,pad_w)))
    np_arr = np.zeros((n_num, w_num, h_num, c_num, block_size), dtype)
    # for n in range(N):
    #     for c in range(C):
    #         for h in range(H):
    #             for w in range(W):
    #                 addr = (c % c_group) * h_group * w_group + (h % h_group) * w_group + (w % w_group)
    #                 if len(activation.shape) == 2:
    #                     np_arr[n, w // w_group, h // h_group, c // c_group, addr] = activation[n, c]
    #                 elif len(activation.shape) == 1:
    #                     np_arr[n, w // w_group, h // h_group, c // c_group, addr] = activation[n]
    #                 else:
    #                     np_arr[n, w // w_group, h // h_group, c // c_group, addr] = activation[n, c, h, w]
    block_size_hacked = 3 * 8 * 8
    c_group_hacked = 3
    for n in range(N):
        for c in range(c_num):
            c_index = c * c_group_hacked
            for h in range(h_num):
                h_index = h * h_group
                for w in range(w_num):
                    w_index = w * w_group
                    # print(activation[n, c_index:c_index+c_group_hacked, h_index:h_index+h_group, w_index:w_index+w_group].shape)
                    np_arr[n, w, h, c, :block_size_hacked] = activation[n, c_index:c_index+c_group_hacked, h_index:h_index+h_group, w_index:w_index+w_group].flatten()
    return np_arr

def build_model(model_path, model_name):

    hash = hashlib.md5()
    hash.update(model_name.encode())
    md5_model = hash.hexdigest()
    model_key = f"{md5_model}:0:{model_name}"
    kwargs = {"name": model_key}

    ctx = tvm.vacc(0)
    loaded_json = open(os.path.join(model_path, model_name) + ".json").read()
    loaded_lib = tvm.module.load(os.path.join(model_path, model_name) + ".so")
    loaded_params = bytearray(open(os.path.join(model_path, model_name) + ".params", "rb").read())
    m = graph_runtime.create(loaded_json, loaded_lib, ctx, **kwargs)  # emu
    m.load_param(loaded_params)
    return m


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def convert_label(label, inverse=False):
    ignore_label = 255
    label_mapping = {-1: ignore_label, 0: ignore_label, 
                            1: ignore_label, 2: ignore_label, 
                            3: ignore_label, 4: ignore_label, 
                            5: ignore_label, 6: ignore_label, 
                            7: 0, 8: 1, 9: ignore_label, 
                            10: ignore_label, 11: 2, 12: 3, 
                            13: 4, 14: ignore_label, 15: ignore_label, 
                            16: ignore_label, 17: 5, 18: ignore_label, 
                            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                            25: 12, 26: 13, 27: 14, 28: 15, 
                            29: ignore_label, 30: ignore_label, 
                            31: 16, 32: 17, 33: 18}
    
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label


if __name__ == '__main__':

    # vacc
    model_path = "deploy_weights/model_final_v2_city0629-int8-kl_divergence-3_736_960-debug"
    model_name = "model_final_v2_city0629"
    input_size = [1, 3, 736, 960]
    data_dir = "/home/simplew/dataset/seg/cityscapes/leftImg8bit/val"
    gt_dir = "/home/simplew/dataset/seg/cityscapes/gtFine/val"

    # torch
    torch_config = "configs/bisenetv2_city.py"
    torch_checkpoint = "model_final_v2_city.pth"
    eval_torch = False


    result_dir = "./vacc_result_vacc"
    os.makedirs(result_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(input_size[2:]),
        transforms.ToTensor(), #->[0,1]
        transforms.Normalize([0.3257, 0.3690, 0.3223], [0.2112, 0.2148, 0.2115])  # ->[-1,1]
    ])
    
    # define torch model
    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../../path/to/CoinCheung/BiSeNet')
    
    from lib.models import model_factory
    from configs import set_cfg_from_file
    cfg = set_cfg_from_file(torch_config)
    torch_model = model_factory[cfg.model_type](cfg.n_cats)
    torch_model.load_state_dict(torch.load(torch_checkpoint, map_location='cpu'))
    torch_model.aux_mode = 'eval'
    torch_model.eval()

    # define vacc model
    vacc_model = build_model(model_path, model_name)
    name = vacc_model.set_batch_size(1)
    
    # eval matrics
    num_classes = 19
    confusion_matrix = np.zeros((num_classes, num_classes))
    colors = np.loadtxt("vacc_code/runmodel/cityscapes_colors.txt").astype('uint8')

    image_files = glob.glob(os.path.join(data_dir, "*/*.png"))
    for image_path in tqdm(image_files):
        image_sub_name = image_path.split("/")[-2] + "/" +  image_path.split("/")[-1]
        # image preprocessing
        ori_image = Image.open(image_path)
        resize_image = ori_image.convert('RGB')
        images = transform(resize_image).unsqueeze(0).numpy()
        
        # model infer
        if eval_torch:
            with torch.no_grad():
                out = torch_model(torch.from_numpy(images))
                heatmap = np.array(out[0])
        else:
            input_image = get_activation_aligned_faster(images.astype("float16"))
            vacc_model.set_input(name, 'input', 0, tvm.nd.array(input_image))
            vacc_model.run(name)

            heatmap = vacc_model.get_output(name, 0, 0).asnumpy().astype("float32")

        tvm_predict = torch.from_numpy(heatmap)
        tvm_predict = F.interpolate(tvm_predict, (input_size[2], input_size[3]), mode='bilinear')     
        
        # draw
        # predict = tvm_predict[0].cpu().numpy()
        # predict = np.asarray(predict, dtype="float32").transpose(1, 2, 0)

        # predict_mask = cv2.resize(predict, ori_image.size, interpolation=cv2.INTER_CUBIC)

        # color = colorize(predict_mask.argmax(axis=2).astype(np.uint8), colors)
        # color.save(os.path.join(result_dir, os.path.basename(image_path)))
        # continue

        ########################################################################################################
        # eval
        label_path = os.path.join(gt_dir, image_sub_name.replace("leftImg8bit.png", "gtFine_labelIds.png"))
        if not os.path.exists(label_path):
            continue
        
        # gt 
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (input_size[3], input_size[2]), interpolation=cv2.INTER_NEAREST)
        label = convert_label(label)
        target = np.array(label).astype('int32')
        target = np.expand_dims(target, 0)
        target = torch.from_numpy(target)

        confusion_matrix += get_confusion_matrix(
            target,
            tvm_predict,
            target.size(),
            num_classes,
            255)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum()/pos.sum()
        mean_acc = (tp/np.maximum(1.0, pos)).mean()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        print("{:s}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(image_path, pixel_acc * 100, mean_IoU * 100))

        # ########################################################################################################

"""
torch
model_final_v2_city.pth
validation pixAcc: 94.713, mIoU: 69.778


model_final_v2_city0629-fp16-none-3_736_960-debug
validation pixAcc: 94.713, mIoU: 69.775

model_final_v2_city0629-int8-kl_divergence-3_736_960-debug
validation pixAcc: 94.510, mIoU: 68.240
"""