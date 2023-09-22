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
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

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
    MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

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

    model_path = "deploy_weights/bisenet-fp16-none-3_512_512-debug"
    model_name = "bisenet"
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
    #     print(heatmap)
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

    classes = 19
    metrics = SegMetric(n_classes=classes)
    metrics.reset()

    for index, (img_path, images, labels) in enumerate(tqdm(data_loader.loader())):
        size = labels.size()
        h, w = size[1], size[2]
        gt = labels.cpu().numpy()
        
        if torch_eval_flag:
            # torch inference
            # torch model
            faceparser = BiSeNet(n_classes = 19)
            faceparser.load_state_dict(torch.load("bisenet.pth", map_location="cpu"))
            faceparser.eval()

            from thop import profile
            from thop import clever_format
            input = torch.randn(1, 3, 512, 512)
            flops, params = profile(faceparser, inputs=(input,))
            print("flops(G):", "%.3f" % (flops / 900000000 * 2))
            flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
            print("params:", params)

            with torch.no_grad():
                _, _, torch_pred = faceparser(images)

            # torch_mask = tenor2mask(torch_pred)[0]
            # cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path[0] + "_torch.png")), torch_mask[:,:,::-1])
            pred = torch_pred.data.max(1)[1].cpu().numpy()
            # parsing = torch_pred.squeeze(0).cpu().numpy().argmax(0)

        else:
            # vacc inference
            input_image = get_activation_aligned_faster(np.array(images).astype("float16"))
            model.set_input(name, 'input', 0, tvm.nd.array(input_image))
            model.run(name)
            heatmap = model.get_output(name, 0, 0).asnumpy().astype("float32")

            vacc_pred = torch.from_numpy(heatmap)
            vacc_mask = tenor2mask(vacc_pred)[0]
            cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path[0] + "_vacc.png")), vacc_mask[:,:,::-1])
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
    facial_names = ['background', 'skin', 'nose', 'eyeglass', 'left_eye', 'right_eye', 'left_brow', 'right_brow',
                    'left_ear', 'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'earring', 'necklace',
                    'neck', 'cloth']
    for i in range(classes):
        print(facial_names[i] + "\t: {}".format(str(class_iou[i])))
    print("--------------------------------------------------------")
    ######################################################################################################

"""
torch 512 classes = 19
----------------- Total Performance --------------------
Overall Acc:     0.9556352229294156
Mean Acc :       0.8337249435210217
FreqW Acc :      0.9164882022332891
Mean IoU :       0.7426032880777604
Overall F1:      0.8410596827641568
----------------- Class IoU Performance ----------------
background      : 0.9378292772392331
skin    : 0.9284966349726388
nose    : 0.6149654277845954
eyeglass        : 0.601570945020349
left_eye        : 0.6525746650178532
right_eye       : 0.6476684443446906
left_brow       : 0.8319301832614219
right_brow      : 0.6705880061787682
left_ear        : 0.6571437967301036
right_ear       : 0.4255793179986799
mouth   : 0.8700320964779847
upper_lip       : 0.8356214172588405
lower_lip       : 0.7608609449202323
hair    : 0.7992254674335968
hat     : 0.8605482739767997
earring : 0.3527505833196731
necklace        : 0.8263593822900838
neck    : 0.9298832924238305
cloth   : 0.9058343168280723



vacc 512 fp16
----------------- Total Performance --------------------
Overall Acc:     0.9556578441703113
Mean Acc :       0.833372987499022
FreqW Acc :      0.916523859125999
Mean IoU :       0.742535004231554
Overall F1:      0.8410015127154231
----------------- Class IoU Performance ----------------
background      : 0.9378759218073169
skin    : 0.9285489511457389
nose    : 0.6146148041426784
eyeglass        : 0.6010793757831302
left_eye        : 0.6523158728168997
right_eye       : 0.6473657791416703
left_brow       : 0.8320022436292658
right_brow      : 0.6705693149717842
left_ear        : 0.6571108468897708
right_ear       : 0.42551992575688297
mouth   : 0.8703116341680609
upper_lip       : 0.8355227523616399
lower_lip       : 0.7607766298012781
hair    : 0.7993262178815398
hat     : 0.8605150492840775
earring : 0.3526342243226829
necklace        : 0.826320220512025
neck    : 0.9299156756029198
cloth   : 0.9058396403801641
--------------------------------------------------------



vacc 512 int8
----------------- Total Performance --------------------
Overall Acc:     0.9556897452579036
Mean Acc :       0.8323962754949473
FreqW Acc :      0.9166093566148232
Mean IoU :       0.7413797567604206
Overall F1:      0.8401734007068263
----------------- Class IoU Performance ----------------
background      : 0.9381407091955711
skin    : 0.9287122321236828
nose    : 0.6082316054863004
eyeglass        : 0.6015556332608095
left_eye        : 0.6488778693748087
right_eye       : 0.6450650386752117
left_brow       : 0.8322012724309388
right_brow      : 0.6648368442550666
left_ear        : 0.6505651232149232
right_ear       : 0.42300831811896916
mouth   : 0.8705586203375313
upper_lip       : 0.8353515645581359
lower_lip       : 0.7602545709801846
hair    : 0.8000063660197961
hat     : 0.8603837252231518
earring : 0.356220163365939
necklace        : 0.8267675595800013
neck    : 0.9300418608525668
cloth   : 0.9054363013944026

"""