# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import argparse
import os
import torch
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
import onnx
from onnxsim import simplify

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    
    #modified config
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


def export_onnx(model, output_dir, model_name):
    # 195
    caption =  'person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . traffic light . fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . broccoli . carrot . hot dog . pizza . donut . cake . chair . couch . potted plant . bed . dining table . toilet . tv . laptop . mouse . remote . keyboard . cell phone . microwave . oven . toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . hair drier . toothbrush .'
    input_ids =  model.tokenizer([caption], return_tensors="pt")["input_ids"]
    position_ids = torch.tensor([[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1,
         2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 2, 3, 0,
         1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
         2, 3, 0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
         1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 2,
         3, 0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0,
         1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0,
         1, 2, 0]])
    
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0]])

    # text_token_mask = np.fromfile("./text_token_mask.npy",dtype=np.bool).reshape(1,195,195)
    # text_token_mask = torch.from_numpy(text_token_mask)

    text_token_mask = torch.randint(0, 2, (1,195,195)).bool()
    
    attention_mask = torch.tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True]])

    # position_ids_np = position_ids.numpy()
    # text_token_mask_np = text_token_mask.numpy()
    # attention_mask_np = attention_mask.numpy()
    # np.save("./195/position_ids.npy", position_ids_np)
    # np.save("./195/text_token_mask.npy",text_token_mask_np)
    # np.save("./195/attention_mask.npy", attention_mask_np)
    # print("ok")
    # exit()


    img = torch.randn(1, 3, 800, 1333)

    #export onnx model
    torch.onnx.export(
    model,
    f=os.path.join(output_dir, model_name+".onnx"),
    args=(img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask), #, zeros, ones),
    input_names=["img" , "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
    output_names=["logits", "boxes"],
    # dynamic_axes=dynamic_axes,
    opset_version=16) # 16
    print(f"onnx export done: {os.path.join(output_dir, model_name+'.onnx')}, simply...")

    model = onnx.load(os.path.join(output_dir, model_name+".onnx"))
    simplified_model, check = simplify(model)
    onnx.save(simplified_model, os.path.join(output_dir, model_name+"_sim.onnx"))
    # onnx.save(simplified_model, "./weights/groundingdino_swint_ogc_img_encoder_sim.onnx", save_as_external_data=True, all_tensors_to_one_file=True)
    print(f"onnx simplify done: {os.path.join(output_dir, model_name+'_sim.onnx')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export Grounding DINO Model to IR", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, default="groundingdino/config/GroundingDINO_SwinT_OGC.py", help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default="weights/groundingdino_swint_ogc.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="weights", help="output directory"
    )
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = args.output_dir
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=True)

    model_name = checkpoint_path.split('/')[-1].split('.')[0]
    
    #export onnx
    export_onnx(model, output_dir, model_name)

###python export_onnx.py -c groundingdino/config/GroundingDINO_SwinT_OGC.py -p weights/groundingdino_swint_ogc.pth -o weights/
###python export_onnx.py -c groundingdino/config/GroundingDINO_SwinB_cfg.py -p weights/groundingdino_swinb_cogcoor.pth -o weights/