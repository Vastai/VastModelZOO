<div id=top align="center">

![logo](./images/index/logo.png)
[![License](https://img.shields.io/badge/license-Apache_2.0-yellow)](LICENSE)
[![company](https://img.shields.io/badge/vastaitech.com-blue)](https://www.vastaitech.com/)


</div>

---

`VastModelZOO`是瀚博半导体维护的AI模型平台，提供了人工智能多个领域（CV、AUDIO、NLP、LLM、MLLM等）的开源模型在瀚博训推芯片上的部署、训练示例。

`VastModelZOO`旨在基于瀚博半导体的硬件产品和软件SDK，展示最佳的编程实践，以达成模型的快速移植和最优性能。

为方便大家使用`VastModelZOO`，我们将持续增加典型模型和基础插件。


## 依赖软件

- 瀚博推理引擎类型：
    - Build_In: 瀚博自研软件栈推理引擎
    - PyTorch: VACC Extension for PyTorch插件

- 版本说明

    |  组件 |    版本    | 
    | :------: | :------: |
    | Driver | 3.1.1 |
    | VastStream | 2.0.T3 |
    | VAMC | 3.4.1 |
    | Pytorch | 2.1.0 |

## 模型列表

<details><summary>CV Models</summary>


- classification

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
|  [CSPNet](./cv/classification/cspnet/README.md)  |   [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/cspnet.py)   |   <details> <summary>model name</summary><ul><li align="left">cspresnet50</li><li align="left">cspresnext50</li><li align="left">cspdarknet53</li></ul></details>   |    classification    |   Build_In    |
|  [CSPNet](./cv/classification/cspnet/README.md)  |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.1/configs/cspnet/README.md)  |   <details> <summary>model name</summary><ul><li align="left">cspresnet50</li><li align="left">cspresnext50</li><li align="left">cspdarknet53</li></ul></details>   |    classification    |   Build_In    |
|  [CSPNet](./cv/classification/cspnet/README.md)  |   [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/Others.md)   |  <details> <summary>model name</summary><ul><li align="left">cspdarknet53</li></ul></details>   |    classification    |   Build_In    |
|  [EfficientNet](./cv/classification/efficientnet/README.md)     |    [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py)    |    <details> <summary>model name</summary><ul><li align="left">efficientnet_b0</li><li align="left">efficientnet_b1</li><li align="left">efficientnet_b2</li><li align="left">efficientnet_b3</li><li align="left">efficientnet_b4</li><li align="left">tf_efficientnet_b5</li><li align="left">tf_efficientnet_b6</li><li align="left">tf_efficientnet_b7</li><li align="left">tf_efficientnet_b8</li></ul></details>    |    classification    |    Build_In    |
|  [EfficientNet_v2](./cv/classification/efficientnet_v2/README.md)  |    [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py)    |  <details> <summary>model name</summary><ul><li align="left">efficientnetv2_rw_t</li><li align="left">efficientnetv2_rw_s</li><li align="left">efficientnetv2_rw_m</li></ul></details>  |    classification    |   Build_In    |
|  [MobileNet_v2](./cv/classification/mobilenet_v2/README.md)     | [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/mobilenetv2.py)  |   <details> <summary>model name</summary><ul><li align="left">mobilenetv2</li></ul></details>   |    classification    |   Build_In    |
|  [MobileNet_v2](./cv/classification/mobilenet_v2/README.md)     |  [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py#L92)  |   <details> <summary>model name</summary><ul><li align="left">mobilenetv2</li></ul></details>   |    classification    |   Build_In    |
|  [MobileNet_v2](./cv/classification/mobilenet_v2/README.md)     |    [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/mobilenet_v2.py)     |   <details> <summary>model name</summary><ul><li align="left">mobilenetv2</li></ul></details>   |    classification    |   Build_In    |
|  [MobileNet_v2](./cv/classification/mobilenet_v2/README.md)     |   [keras](https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/mobilenet_v2.py)   |   <details> <summary>model name</summary><ul><li align="left">mobilenetv2</li></ul></details>   |    classification    |   Build_In    |
|  [MobileNet_v2](./cv/classification/mobilenet_v2/README.md)     |   [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/mobilenet_v2.py)   |   <details> <summary>model name</summary><ul><li align="left">mobilenet_v2_x0.25</li><li align="left">mobilenet_v2_x0.5</li><li align="left">mobilenet_v2_x0.75</li><li align="left">mobilenet_v2</li><li align="left">mobilenet_v2_x1.5</li><li align="left">mobilenet_v2_x2.0</li><li align="left">mobilenet_v2_ssld</li></ul></details>    |    classification    |   Build_In    |
|  [MobileNet_v3](./cv/classification/mobilenet_v3/README.md)     | [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/mobilenetv3.py)  |   <details> <summary>model name</summary><ul><li align="left">mobilenet_v3_small</li><li align="left">mobilenet_v3_large</li></ul></details>    |    classification    |   Build_In    |
|  [MobileNet_v3](./cv/classification/mobilenet_v3/README.md)     |    [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/mobilenetv3.py) |     <details> <summary>model name</summary><ul><li align="left">mobilenet_v3_x1.0</li><li align="left">mobilenet_v3_small_x0.5</li><li align="left">mobilenet_v3_small_x0.75</li><li align="left">mobilenet_v3_small_x1.0</li><li align="left">mobilenet_v3_large_x1.0</li></ul></details> |    classification    |   Build_In    |
|  [MobileNet_v3](./cv/classification/mobilenet_v3/README.md)     |   [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/legendary_models/mobilenet_v3.py)    | <details> <summary>model name</summary><ul><li align="left">mobilenet_v3_small_x0.35</li><li align="left">mobilenet_v3_small_x0.35_ssld</li><li align="left">mobilenet_v3_small_x0.5</li><li align="left">mobilenet_v3_small_x0.75</li><li align="left">mobilenet_v3_small_x1.0</li><li align="left">mobilenet_v3_small_x1.0_ssld</li><li align="left">mobilenet_v3_small_x1.25</li><li align="left">mobilenet_v3_large_x0.35</li><li align="left">mobilenet_v3_large_x0.5</li><li align="left">mobilenet_v3_large_x0.75</li><li align="left">mobilenet_v3_large_x1.0</li><li align="left">mobilenet_v3_large_x1.0_ssld</li><li align="left">mobilenet_v3_large_x1.25</li></ul></details> |    classification    |   Build_In    |
|  [MobileNet_v3](./cv/classification/mobilenet_v3/README.md)     | [showlo](https://github.com/ShowLo/MobileNetV3) | <details> <summary>model name</summary><ul><li align="left">mobilenet_v3_small</li></ul></details>  |    classification    |   Build_In    |
|  [MobileNet_v3](./cv/classification/mobilenet_v3/README.md)     |   [sqlai](https://github.com/xiaolai-sqlai/mobilenetv3)   |   <details> <summary>model name</summary><ul><li align="left">mobilenet_v3_small</li><li align="left">mobilenet_v3_large</li></ul></details>    |    classification    |   Build_In    |
|  [RepOPT](./cv/classification/repopt/README.md)  |  [official](https://github.com/DingXiaoH/RepOptimizers)   |  <details> <summary>model name</summary><ul><li align="left">RepOpt-VGG-B1</li><li align="left">RepOpt-VGG-B2</li><li align="left">RepOpt-VGG-L1</li><li align="left">RepOpt-VGG-L2</li></ul></details>   |    classification    |   Build_In    |
|  [ResNeSt](./cv/classification/resnest/README.md) |   [official](https://github.com/zhanghang1989/ResNeSt)    |    <details> <summary>model name</summary><ul><li align="left">resnest50</li><li align="left">resnest101</li><li align="left">resnest200</li><li align="left">resnest269</li></ul></details>    |    classification    |   Build_In    |
|  [ResNet](./cv/classification/resnet/README.md)  |   [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)   |   <details> <summary>model name</summary><ul><li align="left">resnet18</li><li align="left">resnet26</li><li align="left">resnet34</li><li align="left">resnet50</li><li align="left">resnet101</li><li align="left">resnet152</li><li align="left">gluon_resnet18_v1b</li><li align="left">gluon_resnet34_v1b</li><li align="left">gluon_resnet50_v1b</li><li align="left">gluon_resnet50_v1c</li><li align="left">gluon_resnet50_v1d</li><li align="left">gluon_resnet50_v1s</li><li align="left">gluon_resnet101_v1b</li><li align="left">gluon_resnet101_v1c</li><li align="left">gluon_resnet101_v1d</li><li align="left">gluon_resnet101_v1s</li><li align="left">gluon_resnet152_v1b</li><li align="left">gluon_resnet152_v1c</li><li align="left">gluon_resnet152_v1d</li><li align="left">gluon_resnet152_v1s</li></ul></details>    |    classification    |   Build_In    |
|  [ResNet](./cv/classification/resnet/README.md)  |    [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/resnet.py)    |  <details> <summary>model name</summary><ul><li align="left">resnet18</li><li align="left">resnet34</li><li align="left">resnet50</li><li align="left">resnet101</li><li align="left">resnet152</li></ul></details> |    classification    |   Build_In    |
|  [ResNet](./cv/classification/resnet/README.md)  |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.2/configs/resnet/README.md)  |  <details> <summary>model name</summary><ul><li align="left">resnet18</li><li align="left">resnet34</li><li align="left">resnet50</li><li align="left">resnet101</li><li align="left">resnet152</li></ul></details> |    classification    |   Build_In    |
|  [ResNet](./cv/classification/resnet/README.md)  |   [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/ResNet.md)   |  <details> <summary>model name</summary><ul><li align="left">resnet18</li><li align="left">resnet18_vd</li><li align="left">resnet34</li><li align="left">resnet34_vd</li><li align="left">resnet34_vd_ssld</li><li align="left">resnet50</li><li align="left">resnet50_vc</li><li align="left">resnet50_vd</li><li align="left">resnet50_vd_ssld</li><li align="left">resnet101</li><li align="left">resnet101_vd</li><li align="left">resnet101_vd_ssld</li><li align="left">resnet152</li><li align="left">resnet152_vd</li><li align="left">resnet200_vd</li></ul></details>  |    classification    |   Build_In    |
|  [ResNet](./cv/classification/resnet/README.md)  |  [keras](https://github.com/keras-team/keras/blob/2.3.1/keras/applications/resnet.py)   |     <details> <summary>model name</summary><ul><li align="left">resnet50</li><li align="left">resnet50v2</li><li align="left">resnet101</li><li align="left">resnet101v2</li><li align="left">resnet152</li><li align="left">resnet152v2</li></ul></details> |    classification    |   Build_In    |
|  [ResNet](./cv/classification/resnet/README.md)  | [oneflow](https://github.com/Oneflow-Inc/vision/blob/main/flowvision/models/resnet.py)  |  <details> <summary>model name</summary><ul><li align="left">resnet18</li><li align="left">resnet34</li><li align="left">resnet50</li><li align="left">resnet101</li><li align="left">resnet152</li></ul></details> |    classification    |   Build_In    |
|  [ResNeXt](./cv/classification/resnext/README.md) |   [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)   |     <details> <summary>model name</summary><ul><li align="left">resnext50_32x4d</li><li align="left">resnext50d_32x4d</li><li align="left">resnext101_32x8d</li><li align="left">resnext101_64x4d</li><li align="left">tv_resnext50_32x4d</li></ul></details>     |    classification    |   Build_In    |
|  [ResNeXt](./cv/classification/resnext/README.md) |    [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/resnet.py)    | <details> <summary>model name</summary><ul><li align="left">resnext50_32x4d</li><li align="left">resnext101_32x8d</li></ul></details> |    classification    |   Build_In    |
|  [ResNeXt](./cv/classification/resnext/README.md) | [mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.2/configs/resnext/README.md)  | <details> <summary>model name</summary><ul><li align="left">resnext50_32x4d</li><li align="left">resnext101_32x4d</li><li align="left">resnext101_32x8d</li><li align="left">resnext152_32x4d</li></ul></details> |    classification    |   Build_In    |
|  [ResNeXt](./cv/classification/resnext/README.md) |    [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)     |   <details> <summary>model name</summary><ul><li align="left">resnext50_32x4d</li><li align="left">resnext50_64x4d</li><li align="left">resnext50_vd_32x4d</li><li align="left">resnext50_vd_64x4d</li><li align="left">resnext101_32x4d</li><li align="left">resnext101_64x4d</li><li align="left">resnext101_vd_32x4d</li><li align="left">resnext101_vd_64x4d</li><li align="left">resnext152_32x4d</li><li align="left">resnext152_64x4d</li><li align="left">resnext152_vd_32x4d</li><li align="left">resnext152_vd_64x4d</li><li align="left">resnext101_32x8d_wsl</li><li align="left">resnext101_32x16d_wsl</li><li align="left">resnext101_32x32d_wsl</li></ul></details>    |    classification    |   Build_In    |

- object detection

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [yolov10](./cv/detection/yolov10/README.md) | [yolov10](https://github.com/THU-MIG/yolov10.git) |  <details> <summary>model name</summary><ul><li align="left">YOLOv10-N</li><li align="left">YOLOv10-S</li><li align="left">YOLOv10-M</li><li align="left">YOLOv10-B</li><li align="left">YOLOv10-L</li><li align="left">YOLOv10-X</li></ul></details> |  object detection   |  Build_In | 
| [yolov8](./cv/detection/yolov8/README.md) | [yolov8](https://github.com/ultralytics/ultralytics) |  <details> <summary>model name</summary><ul><li align="left">YOLOv8n</li><li align="left">YOLOv8s</li><li align="left">YOLOv8m</li><li align="left">YOLOv8b</li><li align="left">YOLOv8l</li><li align="left">YOLOv8x</li></ul></details> |  object detection   |  Build_In | 
| [yolov7](./cv/detection/yolov7/README.md) | [yolov7](https://github.com/WongKinYiu/yolov7) |  <details> <summary>model name</summary><ul><li align="left">YOLOv7</li><li align="left">YOLOv7x</li><li align="left">YOLOv7-w6</li><li align="left">YOLOv7-e6</li><li align="left">YOLOv7-d6</li><li align="left">YOLOv7-e6e</li></ul></details> |  object detection   |  Build_In |
| [yolov6](./cv/detection/yolov6/README.md) | [yolov6](https://github.com/meituan/YOLOv6) |  <details> <summary>model name</summary><ul><li align="left">YOLOv6-n</li><li align="left">YOLOv6-tiny</li><li align="left">YOLOv6-s</li></details> |  object detection   |  Build_In |  
| [Yolov5](./cv/detection/yolov5/README.md)  |  [pytorch(u)](https://github.com/ultralytics/yolov5/tree/v6.1)   | <details> <summary>model name</summary><ul><li align="left">yolov5n</li><li align="left">yolov5s</li><li align="left">yolov5m</li><li align="left">yolov5l</li><li align="left">yolov5x</li><li align="left">yolov5n6</li><li align="left">yolov5s6</li><li align="left">yolov5m6</li><li align="left">yolov5l6</li><li align="left">yolov5x6</li></ul></details> |   object detection   |   Build_In  |
| [Yolov5](./cv/detection/yolov5/README.md)  |  [mmyolo](https://github.com/open-mmlab/mmyolo/tree/v0.1.3/configs/yolov5)   | <details> <summary>model name</summary><ul><li align="left">yolov5n</li><li align="left">yolov5s</li><li align="left">yolov5m</li><li align="left">yolov5l</li><li align="left">yolov5n6</li><li align="left">yolov5s6</li><li align="left">yolov5m6</li><li align="left">yolov5l6</li></ul></details>  |   object detection   |  Build_In   |
|  [Yolov4](./cv/detection/yolov4/README.md)   | [darknet](https://github.com/AlexeyAB/darknet)  |  <details> <summary>model name</summary><ul><li align="left">yolov4</li><li align="left">yolov4_tiny</li><li align="left">yolov4_csp</li><li align="left">yolov4_csp_swish</li><li align="left">yolov4_csp_x_swish</li><li align="left">yolov4x_mish</li></ul></details>  |   object detection   |  Buid_In    |
|  [Yolov4](./cv/detection/yolov4/README.md)   | [bubbliiiing](https://github.com/bubbliiiing/yolov4-pytorch)    |    <details> <summary>model name</summary><ul><li align="left">yolov4</li><li align="left">yolov4_tiny</li></ul></details>    |   object detection   |  Buid_In   |
|  [Yolov4](./cv/detection/yolov4/README.md)   | [tianxiaomo](https://github.com/Tianxiaomo/pytorch-YOLOv4) | <details> <summary>model name</summary><ul><li align="left">yolov4</li></ul></details>  |   object detection   |   Buid_In    |
|  [Yolov3](./cv/detection/yolov3/README.md)   |  [pytorch(u)](https://github.com/ultralytics/yolov3/tree/v9.5.0)  | <details> <summary>model name</summary><ul><li align="left">yolov3</li><li align="left">yolov3-spp</li><li align="left">yolov3-tiny</li></ul></details> |   object detection   |   Buid_In    |


- segmentation

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: | 
|    [FCN](./cv/segmentation/fcn/README.md)    |    [pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/README.md) |  <details> <summary>model name</summary><ul><li align="left">fcn8s</li><li align="left">fcn16s</li><li align="left">fcn32s</li></ul></details>  |     segmentation     | Build_In | 
|    [FCN](./cv/segmentation/fcn/README.md)    | [mmseg](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn) |   <details> <summary>model name</summary><ul><li align="left">fcn_r50_d8_20k</li><li align="left">fcn_r50_d8_40k</li><li align="left">fcn_r101_d8_20k</li><li align="left">fcn_r101_d8_40k</li></ul></details>    |     segmentation     | Build_In |
|   [Unet](./cv/segmentation/unet/README.md)   |    [bubbliiiing](https://github.com/bubbliiiing/unet-pytorch) | <details> <summary>model name</summary><ul><li align="left">unet_vgg16</li><li align="left">unet_resnet50</li></ul></details> |     segmentation     | Build_In |
|   [Unet](./cv/segmentation/unet/README.md)   |   [milesial](https://github.com/milesial/Pytorch-UNet)    |   <details> <summary>model name</summary><ul><li align="left">unet_scale0.5</li><li align="left">unet_scale1.0</li></ul></details>    |     segmentation     | Build_In | 
|   [Unet](./cv/segmentation/unet/README.md)   |    [keras](https://github.com/zhixuhao/unet)    |  <details> <summary>model name</summary><ul><li align="left">unet</li></ul></details>   |     segmentation     | Build_In | 
| [UnetPP](./cv/segmentation/unetpp/README.md) |   [pytorch](https://github.com/Andy-zhujunwen/UNET-ZOO)   | <details> <summary>model name</summary><ul><li align="left">unetpp</li></ul></details>  |     segmentation     | Build_In | 
|  [Unet3P](./cv/segmentation/unet3p/README.md) |   [pytorch](https://github.com/avBuffer/UNet3plus_pth)    |  <details> <summary>model name</summary><ul><li align="left">unet3p</li><li align="left">unet3p_deepsupervision</li></ul></details>   |     segmentation     | Build_In | 
|   [Deeplab_v3](./cv/segmentation/deeplab_v3/README.md)   | [pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) |   <details> <summary>model name</summary><ul><li align="left">deeplabv3_resnet50</li><li align="left">deeplabv3_resnet101</li></ul></details>   |     segmentation     | Build_In | 
|     [Deeplab_v3_plus](./cv/segmentation/deeplab_v3/README.md) | [pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) |   <details> <summary>model name</summary><ul><li align="left">deeplabv3plus_resnet50</li><li align="left">deeplabv3plus_resnet101</li></ul></details>   |     segmentation     | Build_In | 
|   [Yolov8-seg](./cv/segmentation/yolov8_seg/README.md)   |  [ultralytics](https://github.com/ultralytics/ultralytics/tree/main)  |     <details> <summary>model name</summary><ul><li align="left">yolov8n-seg</li><li align="left">yolov8s-seg</li><li align="left">yolov8m-seg</li><li align="left">yolov8l-seg</li><li align="left">yolov8x-seg</li></ul></details>     |    instance segmentation | Build_In |
|    [Human_Seg](./cv/segmentation/human_seg/README.md)    | [pytorch](https://github.com/thuyngch/Human-Segmentation-PyTorch) |    <details> <summary>model name</summary><ul><li align="left">unet_resnet18</li><li align="left">deeplabv3plus_resnet18</li></ul></details>    |  human segmentation  | Build_In |
| [MODNet](./cv/segmentation/modnet/README.md) |  [official](https://github.com/ZHKKKe/MODNet)   | <details> <summary>model name</summary><ul><li align="left">modnet</li></ul></details>  |  matting   | Build_In | 
|  [BiSeNet](./cv/segmentation/bisenet/README.md)  |   [pytorch](https://github.com/zllrunning/face-parsing.PyTorch)   |  <details> <summary>model name</summary><ul><li align="left">bisenet</li><li align="left">bisenet_2class</li></ul></details>  |  face segmentation   | Build_In |
|  [BiSeNet](./cv/segmentation/bisenet/README.md)  |     [pytorch](https://github.com/CoinCheung/BiSeNet/)     |   <details> <summary>model name</summary><ul><li align="left">bisenetv1</li><li align="left">bisenetv2</li></ul></details>    |     segmentation     | Build_In | 


</details>

<details><summary>NLP Models</summary>

- Text2Vec

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [bge](./nlp/text2vec/bge/README.md) | [huggingface](https://huggingface.co/BAAI) |  <details> <summary>model name</summary><ul><li align="left">bge-m3</li><li align="left">bge-small-en-v1.5</li><li align="left">bge-base-en-v1.5</li><li align="left">bge-large-en-v1.5</li><li align="left">bge-small-zh-v1.5</li><li align="left">bge-base-zh-v1.5</li><li align="left">bge-large-zh-v1.5</li></ul></details> | Embedding model  |  Build_In | 
| [bce](./nlp/text2vec/bce/README.md) | [huggingface](https://huggingface.co/maidalun1020/bce-embedding-base_v1) |  <details> <summary>model name</summary><ul><li align="left">bce-embedding-base_v1</li></ul></details> | Embedding model  |  Build_In |

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [bge]((./nlp/text2vec/bge/README.md)) | [huggingface](https://huggingface.co/BAAI/) |  <details> <summary>model name</summary><ul><li align="left">bge-reranker-base</li><li align="left">bge-reranker-large</li><li align="left">bge-reranker-v2-m3</li></ul></details> | Reranker model  |  Build_In |
| [bce](./nlp/text2vec/bce/README.md) | [huggingface](https://huggingface.co/maidalun1020/bce-reranker-base_v1) |  <details> <summary>model name</summary><ul><li align="left">bce-reranker-base_v1</li></ul></details> | Reranker model  |  Build_In |

</details>


<details><summary>LLM Models</summary>

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [LLaMA](./llm/llama/README.md) |   [huggingface](https://huggingface.co/meta-llama)    |   <details> <summary>model name</summary><ul><li align="left">meta-llama-33b</li></ul></details>   | large language model |   Build_In/PyTorch   |
| [LLaMA-2](./llm/llama2/README.md) |   [huggingface](https://huggingface.co/meta-llama)    |   <details> <summary>model name</summary><ul><li align="left">Llama-2-7b-hf</li><li align="left">Llama-2-7b-chat-hf</li><li align="left">Llama-2-13b-hf</li><li align="left">Llama-2-13b-chat-hf</li><li align="left">Llama-2-70b-hf</li><li align="left">Llama-2-70b-chat-hf</li></ul></details>   | large language model |   Build_In/PyTorch   |
| [LLaMA-3](./llm/llama3/README.md) |   [huggingface](https://huggingface.co/meta-llama)    |   <details> <summary>model name</summary><ul><li align="left">Meta-Llama-3-8B</li><li align="left">Meta-Llama-3-8B-Instruct</li><li align="left">Meta-Llama-3-70B</li><li align="left">Meta-Llama-3-70B-Instruct</li></ul></details>   | large language model |   Build_In/PyTorch   |
| [LLaMA-3.1](./llm/llama3/README.md) |   [huggingface](https://huggingface.co/meta-llama)    |   <details> <summary>model name</summary><ul><li align="left">Meta-Llama-3.1-8B</li><li align="left">Meta-Llama-3.1-8B-Instruct</li><li align="left">Meta-Llama-3.1-70B</li><li align="left">Meta-Llama-3.1-70B-Instruct</li></ul></details>   | large language model |   Build_In/PyTorch   |
| [LLaMA-3.2](./llm/llama3/README.md) |   [huggingface](https://huggingface.co/meta-llama)    |   <details> <summary>model name</summary><ul><li align="left">Llama-3.2-1B</li><li align="left">Llama-3.2-1B-Instruct</li><li align="left">Llama-3.2-3B</li><li align="left">Llama-3.2-3B-Instruct</li></details>   | large language model |   Build_In/PyTorch   |
| [LLaMA-3.3](./llm/llama3/README.md) |   [huggingface](https://huggingface.co/meta-llama)    |   <details> <summary>model name</summary><ul><li align="left">Llama-3.3-70B-Instruct</li></ul></details>   | large language model |   Build_In/PyTorch   |
| [Qwen1.5](./llm/qwen1.5/README.md) |   [huggingface](https://huggingface.co/Qwen)    |   <details> <summary>model name</summary><ul><li align="left">Qwen1.5-0.5B</li><li align="left">Qwen1.5-0.5B-Chat</li><li align="left">Qwen1.5-1.8B</li><li align="left">Qwen1.5-1.8B-Chat</li><li align="left">Qwen1.5-4B</li><li align="left">Qwen1.5-4B-Chat</li><li align="left">Qwen1.5-7B</li><li align="left">Qwen1.5-7B-Chat</li><li align="left">Qwen1.5-14B</li><li align="left">Qwen1.5-14B-Chat</li><li align="left">Qwen1.5-32B</li><li align="left">Qwen1.5-32B-Chat</li><li align="left">Qwen1.5-72B</li><li align="left">Qwen1.5-72B-Chat</li><li align="left">Qwen1.5-110B-Chat</li></ul></details>   | large language model |   Build_In/PyTorch   |
| [Qwen2](./llm/qwen2/README.md) |   [huggingface](https://huggingface.co/Qwen)    |   <details> <summary>model name</summary><ul><li align="left">Qwen2-0.5B</li><li align="left">Qwen2-0.5B-Instruct</li><li align="left">Qwen2-1.5B</li><li align="left">Qwen2-1.5B-Instruct</li><li align="left">Qwen2-7B</li><li align="left">Qwen2-7B-Instruct</li><li align="left">Qwen2-72B</li><li align="left">Qwen2-72B-Instruct</li></ul></details>   | large language model |   Build_In/PyTorch   |
| [Qwen2.5](./llm/qwen2/README.md) |   [huggingface](https://huggingface.co/Qwen)    |   <details> <summary>model name</summary><ul><li align="left">Qwen2.5-0.5B</li><li align="left">Qwen2.5-0.5B-Instruct</li><li align="left">Qwen2.5-1.5B</li><li align="left">Qwen2.5-1.5B-Instruct</li><li align="left">Qwen2.5-3B</li><li align="left">Qwen2.5-3B-Instruct</li><li align="left">Qwen2.5-7B</li><li align="left">Qwen2.5-7B-Instruct</li><li align="left">Qwen2.5-14B</li><li align="left">Qwen2.5-14B-Instruct</li><li align="left">Qwen2.5-32B</li><li align="left">Qwen2.5-32B-Instruct</li><li align="left">Qwen2.5-72B</li><li align="left">Qwen2.5-72B-Instruct</li></ul></details>   | large language model |   Build_In/PyTorch   |
| [QWQ](./llm/qwq/README.md) |   [huggingface](https://huggingface.co/Qwen)    |   <details> <summary>model name</summary><ul><li align="left">QwQ-32B-Preview</li><li align="left">QwQ-32B</li></ul></details>   | large language model |   Build_In/PyTorch   |
| [DeepSeek-R1-Distill](./llm/deepseek_r1/README.md) |   [huggingface](https://huggingface.co/deepseek-ai)    |   <details> <summary>model name</summary><ul><li align="left">DeepSeek-R1-Distill-Qwen-1.5B</li><li align="left">DeepSeek-R1-Distill-Qwen-7B</li><li align="left">DeepSeek-R1-Distill-Qwen-14B</li><li align="left">DeepSeek-R1-Distill-Qwen-32B</li><li align="left">DeepSeek-R1-Distill-Llama-8B</li><li align="left">DeepSeek-R1-Distill-Llama-70B</li></ul></details>   | large language model |   Build_In/PyTorch   |
| [DeepSeek-V3](./llm/deepseek_v3/README.md) |   [huggingface](https://huggingface.co/deepseek-ai)    |   <details> <summary>model name</summary><ul><li align="left">DeepSeek-V3-Base</li><li align="left">DeepSeek-V3</li><li align="left">DeepSeek-V3-0324</li></ul></details>   | large language model |   vLLM   |
| [DeepSeek-R1](./llm/deepseek_r1/README.md) |   [huggingface](https://huggingface.co/deepseek-ai)    |   <details> <summary>model name</summary><ul><li align="left">DeepSeek-R1</li></ul></details>   | large language model |   vLLM   |

</details>

## 声明

- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集来完成示例，请您特别注意应遵守对应数据集合模型的License，如您因使用数据集或者模型而产生侵权纠纷，瀚博半导体不承担任何责任。
- 如您在使用本地代码的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。