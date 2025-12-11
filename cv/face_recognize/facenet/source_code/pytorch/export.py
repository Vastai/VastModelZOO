# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch
import torchvision
import numpy as np
import pandas as pd
from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
from PIL import Image


if __name__ == '__main__':

    image_path = "./facenet_pytorch/data/test_images_aligned/angelina_jolie/1.png"
    image_size = (160, 160)

    mtcnn = MTCNN()
    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    # resnet = InceptionResnetV1(pretrained='casia-webface').eval()


    img = Image.open(image_path)
    my_preprocess = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(image_size),
                        torchvision.transforms.PILToTensor(),
                        # torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1] do not support PIL
                    ]
                )
    img_data = my_preprocess(img)
    img_data = (img_data - 127.5) / 128.0

    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)

    #### convert onnx
    output = "facenet_vggface2_torch.onnx"
    output = "facenet_casia-webface_torch.onnx"

    input_shape = (1, 3, 160, 160)
    input = torch.randn(*input_shape)
    torch.onnx.export(resnet, input, output, input_names=["input"], output_names= ["output"], export_params=True, opset_version=10)
    scripted_model = torch.jit.trace(resnet, input, strict=False)
    torch.jit.save(scripted_model, output.replace(".onnx", ".torchscript.pt"))

    # Calculate embedding (unsqueeze to add batch dimension)
    embeddings = resnet(img_cropped.unsqueeze(0))
    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    print(pd.DataFrame(dists, columns=["angelina_jolie"], index=["angelina_jolie"]))

    # Or, if using for VGGFace2 classification
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0))
    print(img_probs)