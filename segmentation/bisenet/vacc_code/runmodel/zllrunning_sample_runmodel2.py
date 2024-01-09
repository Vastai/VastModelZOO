import argparse
import glob
import hashlib
import os
import cv2
import numpy as np
import torch
import tvm
from PIL import Image
from torchvision import transforms
from tvm.contrib import graph_runtime
from tqdm import tqdm
import vacc


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

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [255, 0, 0]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'_onehot.png', vis_parsing_anno)
        cv2.imwrite(save_path[:-4] +'_parsing.png', vis_parsing_anno_color)
        cv2.imwrite(save_path[:-4] +'_vis.png', vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_parsing_anno, vis_im

def tenor2mask(tensor_data):
    MASK_COLORMAP = [[0, 0, 0], [0, 0, 255]]

    if len(tensor_data.shape) < 4:
        tensor_data = tensor_data.unsqueeze(0)
    if tensor_data.shape[1] > 1:
        tensor_data = tensor_data.argmax(dim=1) 

    tensor_data = tensor_data.squeeze(1).data.cpu().numpy()
    color_maps = []
    for t in tensor_data:
        tmp_img = np.zeros(tensor_data.shape[1:] + (3,))
        # tmp_img = np.zeros(tensor_data.shape[1:])
        for idx, color in enumerate(MASK_COLORMAP):
            tmp_img[t == idx] = color
        color_maps.append(tmp_img.astype(np.uint8))
    return color_maps


if __name__ == '__main__':

    model_path = "deploy_weights/bisenet_2class_quchu_cloth-fp16-none-3_512_512-debug"
    model_name = "bisenet_2class_quchu_cloth"
    input_size = 512
    data_dir = "/home/simplew/dataset/sr/CelebAMask-HQ/bisegnet_test_img"
    
    save_dir = "./vacc_result"
    os.makedirs(save_dir, exist_ok=True)

    model = build_model(model_path, model_name)

    transform = transforms.Compose([
        transforms.ToTensor(), #->[0,1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ->[-1,1]
    ])

    name =model.set_batch_size(1)

    ###################################vacc infer#######################################
    # for file in glob.glob(os.path.join(data_dir, "*.jpg")):
    #     ori_image = Image.open(file)
    #     resize_image = ori_image.resize((input_size, input_size), Image.BILINEAR).convert('RGB')
    #     images = transform(resize_image).unsqueeze(0).numpy()
    #     input_image = get_activation_aligned_faster(images.astype("float16"))
    #     model.set_input(name, 'input', 0, tvm.nd.array(input_image))
    #     model.run(name)

    #     heatmap = model.get_output(name, 0, 0).asnumpy()
        
    #     # draw image refences to source code
    #     out = torch.from_numpy(heatmap)[0]
    #     parsing = out.squeeze(0).cpu().numpy().argmax(0)
    #     vis_parsing_maps(resize_image, parsing, stride=1, save_im=True, save_path=os.path.join(save_dir, os.path.basename(file)))
    #     # print(heatmap)
    # exit(0)
    ###################################################################################



    ###################################vacc eval#######################################

    torch_eval_flag = False
    import sys
    _cur_file_path = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(_cur_file_path + os.sep + '../..')
    from source_code.face_parsing.eval.data_loader import CustomDataLoader
    from source_code.face_parsing.eval.metrics import SegMetric
    from source_code.face_parsing.model import BiSeNet
    

    data_loader = CustomDataLoader(
        img_path="/home/simplew/dataset/sr/CelebAMask-HQ/bisegnet_test_img",
        label_path="/home/simplew/dataset/sr/CelebAMask-HQ/bisegnet_test_mask",
        image_size=input_size,
        batch_size=1,
        num_workers=0,
        transform=None,
        mode=False)

    classes = 2
    metrics = SegMetric(n_classes=classes)
    metrics.reset()

    for index, (img_path, images, labels) in enumerate(tqdm(data_loader.loader())):
        size = labels.size()
        h, w = size[1], size[2]
        gt = labels.cpu().numpy()
        # wen add
        gt[gt == 16] = 0 # cloth
        gt[gt > 1] = 1  # other merge to one class

        ori_image = Image.open(img_path[0])
        resize_image = ori_image.resize((input_size, input_size), Image.BILINEAR).convert('RGB')

        if torch_eval_flag:
            # torch inference
            # torch model
            faceparser = BiSeNet(n_classes = classes)
            faceparser.load_state_dict(torch.load("79999_iter.pth", map_location="cpu"))
            faceparser.eval()

            with torch.no_grad():
                torch_pred = faceparser(images)
            
            # draw
            # parsing = torch_pred.squeeze(0).cpu().numpy().argmax(0)
            # vis_parsing_maps(resize_image, parsing, stride=1, save_im=True, save_path=os.path.join(save_dir, os.path.basename(img_path[0])))
            
            pred = torch_pred.data.max(1)[1].cpu().numpy()

        else:
            # vacc inference
            input_image = get_activation_aligned_faster(np.array(images).astype("float16"))
            model.set_input(name, 'input', 0, tvm.nd.array(input_image))
            model.run(name)
            heatmap = model.get_output(name, 0, 0).asnumpy().astype("float32")

            vacc_pred = torch.from_numpy(heatmap)

            # drwa
            # parsing = vacc_pred.squeeze(0).cpu().numpy().argmax(0)
            # vis_parsing_maps(resize_image, parsing, stride=1, save_im=True, save_path=os.path.join(save_dir, os.path.basename(img_path[0])))

            pred = vacc_pred.data.max(1)[1].cpu().numpy()
        
        # eval metrics
        try:
            metrics.update(gt, pred)
        except:
            continue
    
    score = metrics.get_scores()[0]
    class_iou = metrics.get_scores()[1]

    print("----------------- Total Performance --------------------")
    for k, v in score.items():
        print(k, v)

    print("----------------- Class IoU Performance ----------------")
    facial_names = ['background', 'all_in_one_except_cloth']
    for i in range(classes):
        print(facial_names[i] + "\t: {}".format(str(class_iou[i])))
    print("--------------------------------------------------------")
    ######################################################################################################

"""
torch 512 classes = 2
----------------- Total Performance --------------------
Overall Acc:     0.9900743658153341
Mean Acc :       0.9875138774962051
FreqW Acc :      0.9803652727764909
Mean IoU :       0.9767597274974249
Overall F1:      0.9882216812155405
----------------- Class IoU Performance ----------------
background      : 0.9676330876295614
all_in_one_except_cloth : 0.9858863673652883
--------------------------------------------------------

deploy_weights/bisenet_2class_quchu_cloth-fp16-none-3_512_512-debug
----------------- Total Performance --------------------
Overall Acc:     0.9900743026675991
Mean Acc :       0.9875126713101177
FreqW Acc :      0.9803651276458463
Mean IoU :       0.9767595299768057
Overall F1:      0.9882215794861182
----------------- Class IoU Performance ----------------
background      : 0.967632757495317
all_in_one_except_cloth : 0.9858863024582945
--------------------------------------------------------

deploy_weights/bisenet_2class_quchu_cloth-int8-kl_divergence-3_512_512-debug
----------------- Total Performance --------------------
Overall Acc:     0.9900391853510482
Mean Acc :       0.9876534861212527
FreqW Acc :      0.9802999683614817
Mean IoU :       0.9766867605825758
Overall F1:      0.9881842393144329
----------------- Class IoU Performance ----------------
background      : 0.9675407247922888
all_in_one_except_cloth : 0.9858327963728626
--------------------------------------------------------


"""