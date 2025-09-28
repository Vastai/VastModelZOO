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
        - 可通过拉取[docker](./docs/docker/README.md)快速在瀚博硬件产品上进行验证测试

- 版本说明

    |  组件 |    版本    |  工具说明  |
    | :------: | :------: | :------: |
    | Driver | 3.3.0 | PCIe 驱动包  |
    | AI-Release | AI3.0_SP9_0811 | AI 工具包 |
    | VAMC | 3.4.1 | 模型转换工具 |
    | VastPipe | 2.7.3 | 全流程低代码开发框架 |
    | VastStreamX | 2.8.3 | SDK应用API库   |

    > Driver及工具文档下载链接: [Baidu Netdisk](https://pan.baidu.com/s/5Lkb0SUPu7r_VdSH53zZWAA)

## 模型列表

<details><summary>CV Models</summary>


- classification

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
|  [CSPNet](./cv/classification/cspnet/README.md)  |   [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/cspnet.py)   |   <details> <summary>model name</summary><ul><li align="left">cspresnet50</li><li align="left">cspresnext50</li><li align="left">cspdarknet53</li></ul></details>   |    classification    |   Build_In    |
|  [CSPNet](./cv/classification/cspnet/README.md)  |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.1/configs/cspnet/README.md)  |   <details> <summary>model name</summary><ul><li align="left">cspresnet50</li><li align="left">cspresnext50</li><li align="left">cspdarknet53</li></ul></details>   |    classification    |   Build_In    |
|  [CSPNet](./cv/classification/cspnet/README.md)  |   [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/Others.md)   |  <details> <summary>model name</summary><ul><li align="left">cspdarknet53</li></ul></details>   |    classification    |   Build_In    |
|  [DenseNet](./cv/classification/densenet/README.md)    |  [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/densenet.py)  |  <details> <summary>model name</summary><ul><li align="left">densenet121</li><li align="left">densenet161</li><li align="left">densenet169</li><li align="left">densenet201</li><li align="left">densenetblur121d</li></ul></details>   |    classification    |   Build_In   |
|  [DenseNet](./cv/classification/densenet/README.md)    |   [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/densenet.py)   | <details> <summary>model name</summary><ul><li align="left">densenet121</li><li align="left">densenet161</li><li align="left">densenet169</li><li align="left">densenet201</li></ul></details>  |    classification    |   Build_In   |
|  [DenseNet](./cv/classification/densenet/README.md)    | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/densenet.py)  | <details> <summary>model name</summary><ul><li align="left">densenet121</li><li align="left">densenet161</li><li align="left">densenet169</li><li align="left">densenet201</li></ul></details>  |    classification    |   Build_In   |
|  [DenseNet](./cv/classification/densenet/README.md)    | [keras](https://github.com/keras-team/keras/blob/2.3.1/keras/applications/densenet.py)  |    <details> <summary>model name</summary><ul><li align="left">densenet121</li><li align="left">densenet169</li><li align="left">densenet201</li></ul></details>    |    classification    |   Build_In   |
|  [DenseNet](./cv/classification/densenet/README.md)    |   [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#dpn-与-densenet-系列-1415)   |     <details> <summary>model name</summary><ul><li align="left">densenet121</li><li align="left">densenet161</li><li align="left">densenet169</li><li align="left">densenet201</li><li align="left">densenet264</li></ul></details>     |    classification    |   Build_In   |
|  [DLANet](./cv/classification/dlanet/README.md)  |    [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) |  <details> <summary>model name</summary><ul><li align="left">dla34</li><li align="left">dla46_c</li><li align="left">dla46x_c</li><li align="left">dla60</li><li align="left">dla60x</li><li align="left">dla60x_c</li><li align="left">dla102</li><li align="left">dla102x</li><li align="left">dla102x2</li><li align="left">dla169</li><li align="left">dla60_res2net</li><li align="left">dla60_res2next</li></ul></details>  |    classification    |   Build_In   |
|  [DLANet](./cv/classification/dlanet/README.md)  |    [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/DLA.md) |    <details> <summary>model name</summary><ul><li align="left">dla34</li><li align="left">dla46_c</li><li align="left">dla60</li><li align="left">dla60x</li><li align="left">dla60x_c</li><li align="left">dla102</li><li align="left">dla102x</li><li align="left">dla102x2</li><li align="left">dla169</li></ul></details> |    classification    |   Build_In   |
|  [DLANet](./cv/classification/dlanet/README.md)  |  [ucbdrive](https://github.com/ucbdrive/dla/blob/master/dla.py)   | <details> <summary>model name</summary><ul><li align="left">dla34</li><li align="left">dla46_c</li><li align="left">dla46x_c</li><li align="left">dla60</li><li align="left">dla60x</li><li align="left">dla60x_c</li><li align="left">dla102</li><li align="left">dla102x</li><li align="left">dla102x2</li><li align="left">dla169</li></ul></details>  |    classification    |   Build_In   |
|  [EfficientNet](./cv/classification/efficientnet/README.md)     |    [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py)    |    <details> <summary>model name</summary><ul><li align="left">efficientnet_b0</li><li align="left">efficientnet_b1</li><li align="left">efficientnet_b2</li><li align="left">efficientnet_b3</li><li align="left">efficientnet_b4</li><li align="left">tf_efficientnet_b5</li><li align="left">tf_efficientnet_b6</li><li align="left">tf_efficientnet_b7</li><li align="left">tf_efficientnet_b8</li></ul></details>    |    classification    |    Build_In    |
|  [EfficientNet_v2](./cv/classification/efficientnet_v2/README.md)  |    [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py)    |  <details> <summary>model name</summary><ul><li align="left">efficientnetv2_rw_t</li><li align="left">efficientnetv2_rw_s</li><li align="left">efficientnetv2_rw_m</li></ul></details>  |    classification    |   Build_In    |
|  [ESNet](./cv/classification/esnet/README.md) |   [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/legendary_models/esnet.py)   |  <details> <summary>model name</summary><ul><li align="left">ESNet_x0_25</li><li align="left">ESNet_x0_5</li><li align="left">ESNet_x0_75</li><li align="left">ESNet_x1_0</li></ul></details>   |    classification    |   Build_In   |
|  [GhostNet](./cv/classification/ghostnet/README.md)    |  [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/ghostnet.py)  |  <details> <summary>model name</summary><ul><li align="left">ghostnet_100</li></ul></details>   |    classification    |   Build_In   |
|  [GhostNet](./cv/classification/ghostnet/README.md)    |  [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release%2F2.5/docs/zh_CN/models/ImageNet1k/GhostNet.md)  |    <details> <summary>model name</summary><ul><li align="left">ghostnet_x0_5</li><li align="left">ghostnet_x1_0</li><li align="left">ghostnet_x1_3</li><li align="left">ghostnet_x1_3_ssld</li></ul></details>    |    classification    |   Build_In   |
|  [HRNet](./cv/classification/hrnet/README.md) |  [official](https://github.com/HRNet/HRNet-Image-Classification)  |   <details> <summary>model name</summary><ul><li align="left">HRNet_w18_small_model_v1</li><li align="left">HRNet_w18_small_model_v2</li><li align="left">HRNet_w18</li><li align="left">HRNet_w30</li><li align="left">HRNet_w32</li><li align="left">HRNet_w40</li><li align="left">HRNet_w44</li><li align="left">HRNet_w48</li><li align="left">HRNet_w64</li></ul></details>   |    classification    |   Build_In   |
|  [HRNet](./cv/classification/hrnet/README.md) |   [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py)   |  <details> <summary>model name</summary><ul><li align="left">HRNet_w18</li><li align="left">HRNet_w30</li><li align="left">HRNet_w32</li><li align="left">HRNet_w40</li><li align="left">HRNet_w44</li><li align="left">HRNet_w48</li><li align="left">HRNet_w64</li></ul></details>  |    classification    |   Build_In   |
|  [HRNet](./cv/classification/hrnet/README.md) |   [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)    |   <details> <summary>model name</summary><ul><li align="left">HRNet_w18_small_model_v1</li><li align="left">HRNet_w18_small_model_v2</li><li align="left">HRNet_w18</li><li align="left">HRNet_w30</li><li align="left">HRNet_w32</li><li align="left">HRNet_w40</li><li align="left">HRNet_w44</li><li align="left">HRNet_w48</li><li align="left">HRNet_w64</li></ul></details>   |    classification    |   Build_In   |
|  [HRNet](./cv/classification/hrnet/README.md) |     [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/algorithm_introduction/ImageNet_models.md#hrnet-系列-13)     |  <details> <summary>model name</summary><ul><li align="left">HRNet_W18_C</li><li align="left">HRNet_W18_C_ssld</li><li align="left">HRNet_W30_C</li><li align="left">HRNet_W32_C</li><li align="left">HRNet_W40_C</li><li align="left">HRNet_W44_C</li><li align="left">HRNet_W48_C</li><li align="left">HRNet_W48_C_ssld</li><li align="left">HRNet_W64_C</li><li align="left">SE_HRNet_W64_C_ssld</li></ul></details>   |    classification    |   Build_In   |
|  [Inception_v3](./cv/classification/inception_v3/README.md)     |    [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/inception_v3.py)    |  <details> <summary>model name</summary><ul><li align="left">inception_v3</li><li align="left">tf_inception_v3</li><li align="left">adv_inception_v3</li><li align="left">gluon_inception_v3</li></ul></details>  |    classification    |   Build_In   |
|  [Inception_v3](./cv/classification/inception_v3/README.md)     |   [torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py)    |  <details> <summary>model name</summary><ul><li align="left">inception_v3</li></ul></details>   |    classification    |   Build_In   |
|  [Inception_v3](./cv/classification/inception_v3/README.md)     | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/Inception.md)  |  <details> <summary>model name</summary><ul><li align="left">inception_v3</li></ul></details>   |    classification    |   Build_In   |
|  [Inception_v4](./cv/classification/inception_v4/README.md)     |    [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/inception_v4.py)    |  <details> <summary>model name</summary><ul><li align="left">inception v4</li></ul></details>   |    classification    |   Build_In   |
|  [Inception_v4](./cv/classification/inception_v4/README.md)     | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/Inception.md)  |  <details> <summary>model name</summary><ul><li align="left">inception v4</li></ul></details>   |    classification    |   Build_In   |
|  [IResNet](./cv/classification/iresnet/README.md) |  [official](https://github.com/iduta/iresnet/blob/master/models/iresnet.py)   |    <details> <summary>model name</summary><ul><li align="left">iresnet50</li><li align="left">iresnet101</li><li align="left">iresnet152</li><li align="left">iresnet200</li></ul></details>    |    classification    |   Build_In   |
|  [MNasNet](./cv/classification/mnasnet/README.md) |   [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/mnasnet.py)    |  <details> <summary>model name</summary><ul><li align="left">mnasnet0_5</li><li align="left">mnasnet1_0</li></ul></details>   |    classification    |   Build_In   |
|  [MobileNet_v1](./cv/classification/mobilenet_v1/README.md)     |    [keras](https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/mobilenet.py) |   <details> <summary>model name</summary><ul><li align="left">mobilenetv1</li></ul></details>   |    classification    |   Build_In   |
|  [MobileNet_v1](./cv/classification/mobilenet_v1/README.md)     |   [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/legendary_models/mobilenet_v1.py)    |    <details> <summary>model name</summary><ul><li align="left">mobilenet_v1_x0.25</li><li align="left">mobilenet_v1_x0.5</li><li align="left">mobilenet_v1_x0.75</li><li align="left">mobilenet_v1</li><li align="left">mobilenet_v1_ssld</li></ul></details>     |    classification    |   Build_In   |
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
|  [MobileOne](./cv/classification/mobileone/README.md)     |   [official](https://github.com/apple/ml-mobileone/blob/main/mobileone.py)    | <details> <summary>model name</summary><ul><li align="left">mobileone_s0</li><li align="left">mobileone_s1</li><li align="left">mobileone_s2</li><li align="left">mobileone_s3</li><li align="left">mobileone_s4</li></ul></details> |    classification    |   Build_In    |
|  [MobileViT](./cv/classification/mobilevit/README.md) |  [apple](https://github.com/apple/ml-cvnets)   |    <details> <summary>model name</summary><ul><li align="left">mobilevit-s</li></ul></details>    |    classification    |   Build_In   |
|  [PPLCNet](./cv/classification/pplcnet/README.md) | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/legendary_models/pp_lcnet.py)  |    <details> <summary>model name</summary><ul><li align="left">PPLCNet_x0_25</li><li align="left">PPLCNet_x0_35</li><li align="left">PPLCNet_x0_5</li><li align="left">PPLCNet_x0_75</li><li align="left">PPLCNet_x1_0</li><li align="left">PPLCNet_x1_5</li><li align="left">PPLCNet_x2_0</li><li align="left">PPLCNet_x2_5</li><li align="left">PPLCNet_x0_5_ssld</li><li align="left">PPLCNet_x1_0_ssld</li><li align="left">PPLCNet_x2_5_ssld</li></ul></details>     |    classification    |   Build_In   |
|  [PPLCNet_v2](./cv/classification/pplcnet_v2/README.md)  |    [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/legendary_models/pp_lcnet_v2.py)    |     <details> <summary>model name</summary><ul><li align="left">PPLCNetV2_base</li><li align="left">PPLCNetV2_base_ssld</li></ul></details>     |    classification    |   Build_In   |
|  [RepOPT](./cv/classification/repopt/README.md)  |  [official](https://github.com/DingXiaoH/RepOptimizers)   |  <details> <summary>model name</summary><ul><li align="left">RepOpt-VGG-B1</li><li align="left">RepOpt-VGG-B2</li><li align="left">RepOpt-VGG-L1</li><li align="left">RepOpt-VGG-L2</li></ul></details>   |    classification    |   Build_In    |
| [RepVGG](./cv/classification/repvgg/README.md)  | [official](https://github.com/DingXiaoH/RepVGG) |  <details> <summary>model name</summary><ul><li align="left">RepVGG-A0</li><li align="left">RepVGG-A1</li><li align="left">RepVGG-A2</li><li align="left">RepVGG-B0</li><li align="left">RepVGG-B1</li><li align="left">RepVGG-B2</li><li align="left">RepVGG-B1g2</li><li align="left">RepVGG-B1g4</li><li align="left">RepVGG-B2g4</li><li align="left">RepVGG-B3</li><li align="left">RepVGG-B3g4</li></ul></details>  |    classification    |   Build_In   |
|    [Res2Net](./cv/classification/res2net/README.md) |   [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/Res2Net.md)    |  <details> <summary>model name</summary><ul><li align="left">res2net50_14w_8s</li><li align="left">res2net50_26w_4s</li><li align="left">res2net50_vd_26w_4s</li><li align="left">res2net50_vd_26w_4s_ssld</li><li align="left">res2net101_vd_26w_4s</li><li align="left">res2net101_vd_26w_4s_ssld</li><li align="left">res2net200_vd_26w_4s</li><li align="left">res2net200_vd_26w_4s_ssld</li></ul></details>  |    classification    |   Build_In   |
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
| [SENet](./cv/classification/senet/README.md) |   [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)   |   <details> <summary>model name</summary><ul><li align="left">seresnet50</li><li align="left">seresnet152d</li><li align="left">seresnext26d_32x4d</li><li align="left">seresnext26t_32x4d</li><li align="left">seresnext50_32x4d</li><li align="left">seresnext101_32x8d</li><li align="left">seresnext101d_32x8d</li><li align="left">legacy_seresnet18</li><li align="left">legacy_seresnet34</li><li align="left">legacy_seresnet50</li><li align="left">legacy_senet154</li><li align="left">legacy_seresnext26_32x4d</li><li align="left">legacy_seresnext50_32x4d</li><li align="left">legacy_seresnext101_32x4d</li><li align="left">legacy_seresnet101</li><li align="left">legacy_seresnet152</li></ul></details>   |    classification    |   Build_In   |
|  [ShuffleNet_v1](./cv/classification/shufflenet_v1/README.md)    |   [mmcls](https://github.com/open-mmlab/mmclassification/tree/v0.23.1/configs/shufflenet_v1)    |  <details> <summary>model name</summary><ul><li align="left">shufflenet_v1</li></ul></details>  |    classification    |   Build_In   |
|  [ShuffleNet_v1](./cv/classification/shufflenet_v1/README.md)    |  [megvii](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1)   |  <details> <summary>model name</summary><ul><li align="left">Group3_0.5x</li><li align="left">Group3_1.0x</li><li align="left">Group3_1.5x</li><li align="left">Group3_2.0x</li><li align="left">Group8_0.5x</li><li align="left">Group8_1.0x</li><li align="left">Group8_1.5x</li><li align="left">Group8_2.0x</li></ul></details> |    classification    |   Build_In   |
|  [ShuffleNet_v2](./cv/classification/shufflenet_v2/README.md)    | [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/shufflenetv2.py) |   <details> <summary>model name</summary><ul><li align="left">shufflenet_v2_x0_5</li><li align="left">shufflenet_v2_x1_0</li></ul></details>    |    classification    |   Build_In   |
|  [ShuffleNet_v2](./cv/classification/shufflenet_v2/README.md)    |    [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/shufflenet_v2.py)    | <details> <summary>model name</summary><ul><li align="left">shufflenet_v2_x1_0</li></ul></details>  |    classification    |   Build_In   |
|  [ShuffleNet_v2](./cv/classification/shufflenet_v2/README.md)    |  [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)   |  <details> <summary>model name</summary><ul><li align="left">shufflenet_v2_x0_25</li><li align="left">shufflenet_v2_x0_33</li><li align="left">shufflenet_v2_x0_5</li><li align="left">shufflenet_v2_x1_0</li><li align="left">shufflenet_v2_x1_5</li><li align="left">shufflenet_v2_x2_0</li><li align="left">shufflenet_v2_swish</li></ul></details>  |    classification    |   Build_In   |
|  [ShuffleNet_v2](./cv/classification/shufflenet_v2/README.md)    | [megvii](https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/README.md)  |  <details> <summary>model name</summary><ul><li align="left">shufflenet_v2_x0.5</li><li align="left">shufflenet_v2_x1.0</li><li align="left">shufflenet_v2_x1.5</li><li align="left">shufflenet_v2_x2.0</li></ul></details>   |    classification    |   Build_In   |
| [Swin](./cv/classification/swin_transformer/README.md) |  [microsoft](https://github.com/microsoft/Swin-Transformer?tab=readme-ov-file)   |    <details> <summary>model name</summary><ul><li align="left">swin-b</li></ul></details>    |    classification    |   Build_In   |
|   [VGG](./cv/classification/vgg/README.md)   |    [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/vgg.py) | <details> <summary>model name</summary><ul><li align="left">vgg11</li><li align="left">vgg11_bn</li><li align="left">vgg13</li><li align="left">vgg13_bn</li><li align="left">vgg16</li><li align="left">vgg16_bn</li><li align="left">vgg19</li><li align="left">vgg19_bn</li></ul></details>  |    classification    |   Build_In  |
|   [VGG](./cv/classification/vgg/README.md)   | [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)  | <details> <summary>model name</summary><ul><li align="left">vgg11</li><li align="left">vgg11_bn</li><li align="left">vgg13</li><li align="left">vgg13_bn</li><li align="left">vgg16</li><li align="left">vgg16_bn</li><li align="left">vgg19</li><li align="left">vgg19_bn</li></ul></details>  |    classification    |   Build_In  |
|   [VGG](./cv/classification/vgg/README.md)   |   [mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.2/configs/vgg/README.md)    | <details> <summary>model name</summary><ul><li align="left">vgg11</li><li align="left">vgg11_bn</li><li align="left">vgg13</li><li align="left">vgg13_bn</li><li align="left">vgg16</li><li align="left">vgg16_bn</li><li align="left">vgg19</li><li align="left">vgg19_bn</li></ul></details>  |    classification    |   Build_In  |
|   [VGG](./cv/classification/vgg/README.md)   |   [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/VGG.md)    | <details> <summary>model name</summary><ul><li align="left">vgg11</li><li align="left">vgg13</li><li align="left">vgg16</li><li align="left">vgg19</li></ul></details>  |    classification    |   Build_In  |
| [ViT](./cv/classification/vision_transformer/README.md) |  [huggingface](https://huggingface.co/google/vit-base-patch16-224)   |    <details> <summary>model name</summary><ul><li align="left">vit-base-patch16-224</li></ul></details>    |    classification    |   Build_In   |
|  [VOVNet](./cv/classification/vovnet/README.md)  |   [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/vovnet.py)   |  <details> <summary>model name</summary><ul><li align="left">ese_vovnet19b_dw</li><li align="left">ese_vovnet39b</li></ul></details>  |    classification    |   Build_In   |
| [Wide ResNet](./cv/classification/wide_resnet/README.md) |   [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)   | <details> <summary>model name</summary><ul><li align="left">wide_resnet50_2</li><li align="left">wide_resnet101_2</li></ul></details> |    classification    |   Build_In   |
| [Wide ResNet](./cv/classification/wide_resnet/README.md) |    [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/resnet.py)    | <details> <summary>model name</summary><ul><li align="left">wide_resnet50_2</li><li align="left">wide_resnet101_2</li></ul></details> |    classification    |   Build_In   |
|    [Xception](./cv/classification/xception/README.md)    |   [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models/Inception.md)   |   <details> <summary>model name</summary><ul><li align="left">xception41</li><li align="left">xception41_deeplab</li><li align="left">xception65</li><li align="left">xception65_deeplab</li><li align="left">xception71</li></ul></details>    |    classification    |   Build_In   |

- object detection

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [CenterNet](./cv/detection/centernet/README.md)  |    [official](https://github.com/xingyizhou/CenterNet)    | <details> <summary>model name</summary><ul><li align="left">centernet_res18</li></ul></details> |   object detection   | Build_In | 
| [DETR](./cv/detection/detr/README.md)  |   [facebook](https://github.com/facebookresearch/detr?tab=readme-ov-file)   | <details> <summary>model name</summary><ul><li align="left">detr_res50</li></ul></details> |   object detection   | Build_In |
| [Grounding-DINO](./cv/detection/grounding_dino/README.md)  |   [official](https://github.com/IDEA-Research/GroundingDINO)   | <details> <summary>model name</summary><ul><li align="left">groundingdino_swint_ogc</li><li align="left">groundingdino_swinb_cogcoor</li></ul></details> |   object detection   | Build_In |
| [RT-DETR](./cv/detection/rtdetr/README.md)  |   [official](https://github.com/lyuwenyu/RT-DETR)   | <details> <summary>model name</summary><ul><li align="left">rtdetr_r18vd</li></ul></details> |   object detection   | Build_In |
| [YOLO-World](./cv/detection/yolo_world/README.md)  |   [official](https://github.com/AILab-CVC/YOLO-World)   | <details> <summary>model name</summary><ul><li align="left">yolo_world_v2_l</li></ul></details> |   object detection   | Build_In | 
|  [Yolov3](./cv/detection/yolov3/README.md)   |  [pytorch(u)](https://github.com/ultralytics/yolov3/tree/v9.5.0)  | <details> <summary>model name</summary><ul><li align="left">yolov3</li><li align="left">yolov3-spp</li><li align="left">yolov3-tiny</li></ul></details> |   object detection   |   Buid_In    |
|  [Yolov4](./cv/detection/yolov4/README.md)   | [darknet](https://github.com/AlexeyAB/darknet)  |  <details> <summary>model name</summary><ul><li align="left">yolov4</li><li align="left">yolov4_tiny</li><li align="left">yolov4_csp</li><li align="left">yolov4_csp_swish</li><li align="left">yolov4_csp_x_swish</li><li align="left">yolov4x_mish</li></ul></details>  |   object detection   |  Buid_In    |
|  [Yolov4](./cv/detection/yolov4/README.md)   | [bubbliiiing](https://github.com/bubbliiiing/yolov4-pytorch)    |    <details> <summary>model name</summary><ul><li align="left">yolov4</li><li align="left">yolov4_tiny</li></ul></details>    |   object detection   |  Buid_In   |
|  [Yolov4](./cv/detection/yolov4/README.md)   | [tianxiaomo](https://github.com/Tianxiaomo/pytorch-YOLOv4) | <details> <summary>model name</summary><ul><li align="left">yolov4</li></ul></details>  |   object detection   |   Buid_In    |
| [Yolov5](./cv/detection/yolov5/README.md)  |  [pytorch(u)](https://github.com/ultralytics/yolov5/tree/v6.1)   | <details> <summary>model name</summary><ul><li align="left">yolov5n</li><li align="left">yolov5s</li><li align="left">yolov5m</li><li align="left">yolov5l</li><li align="left">yolov5x</li><li align="left">yolov5n6</li><li align="left">yolov5s6</li><li align="left">yolov5m6</li><li align="left">yolov5l6</li><li align="left">yolov5x6</li></ul></details> |   object detection   |   Build_In  |
| [Yolov5](./cv/detection/yolov5/README.md)  |  [mmyolo](https://github.com/open-mmlab/mmyolo/tree/v0.1.3/configs/yolov5)   | <details> <summary>model name</summary><ul><li align="left">yolov5n</li><li align="left">yolov5s</li><li align="left">yolov5m</li><li align="left">yolov5l</li><li align="left">yolov5n6</li><li align="left">yolov5s6</li><li align="left">yolov5m6</li><li align="left">yolov5l6</li></ul></details>  |   object detection   |  Build_In   |
| [yolov6](./cv/detection/yolov6/README.md) | [yolov6](https://github.com/meituan/YOLOv6) |  <details> <summary>model name</summary><ul><li align="left">YOLOv6-n</li><li align="left">YOLOv6-tiny</li><li align="left">YOLOv6-s</li></details> |  object detection   |  Build_In |  
| [yolov7](./cv/detection/yolov7/README.md) | [yolov7](https://github.com/WongKinYiu/yolov7) |  <details> <summary>model name</summary><ul><li align="left">YOLOv7</li><li align="left">YOLOv7x</li><li align="left">YOLOv7-w6</li><li align="left">YOLOv7-e6</li><li align="left">YOLOv7-d6</li><li align="left">YOLOv7-e6e</li></ul></details> |  object detection   |  Build_In |
| [yolov8](./cv/detection/yolov8/README.md) | [yolov8](https://github.com/ultralytics/ultralytics) |  <details> <summary>model name</summary><ul><li align="left">YOLOv8n</li><li align="left">YOLOv8s</li><li align="left">YOLOv8m</li><li align="left">YOLOv8b</li><li align="left">YOLOv8l</li><li align="left">YOLOv8x</li></ul></details> |  object detection   |  Build_In | 
| [yolov10](./cv/detection/yolov10/README.md) | [yolov10](https://github.com/THU-MIG/yolov10.git) |  <details> <summary>model name</summary><ul><li align="left">YOLOv10-N</li><li align="left">YOLOv10-S</li><li align="left">YOLOv10-M</li><li align="left">YOLOv10-B</li><li align="left">YOLOv10-L</li><li align="left">YOLOv10-X</li></ul></details> |  object detection   |  Build_In | 
|  [Yolov11](./cv/detection/yolov11/README.md)   | [official](https://github.com/ultralytics/ultralytics) |    <details> <summary>model name</summary><ul><li align="left">yolo11n</li><li align="left">yolo11s</li><li align="left">yolo11m</li><li align="left">yolo11l</li><li align="left">yolo11x</li></ul></details>    |   object detection   |   Build_In   | 
|  [Yolov12](./cv/detection/yolov12/README.md)   | [official](https://github.com/sunsmarterjie/yolov12.git) |    <details> <summary>model name</summary><ul><li align="left">yolov12n</li><li align="left">yolov12s</li><li align="left">yolov12m</li><li align="left">yolov12l</li><li align="left">yolov12x</li></ul></details>    |   object detection   |   Build_In   |
|   [Yolox](./cv/detection/yolox/README.md)    | [official](https://github.com/Megvii-BaseDetection/YOLOX) |   <details> <summary>model name</summary><ul><li align="left">yolox_s</li><li align="left">yolox_m</li><li align="left">yolox_l</li><li align="left">yolox_x</li><li align="left">yolox_darknet</li><li align="left">yolox_tiny</li><li align="left">yolox_nano</li></ul></details>   |   object detection   |   Build_In   |
|   [Yolox](./cv/detection/yolox/README.md)    |    [mmyolo](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolox/README.md)     |    <details> <summary>model name</summary><ul><li align="left">yolox_s</li><li align="left">yolox_tiny</li></ul></details>    |   object detection   | Build_In |


- segmentation

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: | 
| [BiSeNet](./cv/segmentation/bisenet/README.md)  |     [pytorch](https://github.com/CoinCheung/BiSeNet/)     |   <details> <summary>model name</summary><ul><li align="left">bisenetv1</li><li align="left">bisenetv2</li></ul></details>    |     segmentation     | Build_In | 
| [BiSeNet](./cv/segmentation/bisenet/README.md)  |   [pytorch](https://github.com/zllrunning/face-parsing.PyTorch)   |  <details> <summary>model name</summary><ul><li align="left">bisenet</li><li align="left">bisenet_2class</li></ul></details>  |  face segmentation   | Build_In |
| [Deeplab_v3](./cv/segmentation/deeplab_v3/README.md)   | [pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) |   <details> <summary>model name</summary><ul><li align="left">deeplabv3_resnet50</li><li align="left">deeplabv3_resnet101</li></ul></details>   |     segmentation     | Build_In | 
| [Deeplab_v3_plus](./cv/segmentation/deeplab_v3/README.md) | [pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) |   <details> <summary>model name</summary><ul><li align="left">deeplabv3plus_resnet50</li><li align="left">deeplabv3plus_resnet101</li></ul></details>   |     segmentation     | Build_In | 
| [FCN](./cv/segmentation/fcn/README.md)    |    [pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/README.md) |  <details> <summary>model name</summary><ul><li align="left">fcn8s</li><li align="left">fcn16s</li><li align="left">fcn32s</li></ul></details>  |     segmentation     | Build_In | 
| [FCN](./cv/segmentation/fcn/README.md)    | [mmseg](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn) |   <details> <summary>model name</summary><ul><li align="left">fcn_r50_d8_20k</li><li align="left">fcn_r50_d8_40k</li><li align="left">fcn_r101_d8_20k</li><li align="left">fcn_r101_d8_40k</li></ul></details>    |     segmentation     | Build_In |
| [Human_Seg](./cv/segmentation/human_seg/README.md)    | [pytorch](https://github.com/thuyngch/Human-Segmentation-PyTorch) |    <details> <summary>model name</summary><ul><li align="left">unet_resnet18</li><li align="left">deeplabv3plus_resnet18</li></ul></details>    |  human segmentation  | Build_In |
| [Mask2Former](./cv/segmentation/mask2former/README.md)   |   [official](https://github.com/facebookresearch/Mask2Former)    |  <details> <summary>model name</summary><ul><li align="left">maskformer2_R50</li></ul></details>  |     instance segmentation     | Build_In | 
| [MODNet](./cv/segmentation/modnet/README.md) |  [official](https://github.com/ZHKKKe/MODNet)   | <details> <summary>model name</summary><ul><li align="left">modnet</li></ul></details>  |  matting   | Build_In | 
| [Unet](./cv/segmentation/unet/README.md)   |    [bubbliiiing](https://github.com/bubbliiiing/unet-pytorch) | <details> <summary>model name</summary><ul><li align="left">unet_vgg16</li><li align="left">unet_resnet50</li></ul></details> |     segmentation     | Build_In |
| [Unet](./cv/segmentation/unet/README.md)   |   [milesial](https://github.com/milesial/Pytorch-UNet)    |   <details> <summary>model name</summary><ul><li align="left">unet_scale0.5</li><li align="left">unet_scale1.0</li></ul></details>    |     segmentation     | Build_In | 
| [Unet](./cv/segmentation/unet/README.md)   |    [keras](https://github.com/zhixuhao/unet)    |  <details> <summary>model name</summary><ul><li align="left">unet</li></ul></details>   |     segmentation     | Build_In | 
| [Unet3P](./cv/segmentation/unet3p/README.md) |   [pytorch](https://github.com/avBuffer/UNet3plus_pth)    |  <details> <summary>model name</summary><ul><li align="left">unet3p</li><li align="left">unet3p_deepsupervision</li></ul></details>   |     segmentation     | Build_In | 
| [UnetPP](./cv/segmentation/unetpp/README.md) |   [pytorch](https://github.com/Andy-zhujunwen/UNET-ZOO)   | <details> <summary>model name</summary><ul><li align="left">unetpp</li></ul></details>  |     segmentation     | Build_In | 
| [Yolov8-seg](./cv/segmentation/yolov8_seg/README.md)   |  [ultralytics](https://github.com/ultralytics/ultralytics/tree/main)  |     <details> <summary>model name</summary><ul><li align="left">yolov8n-seg</li><li align="left">yolov8s-seg</li><li align="left">yolov8m-seg</li><li align="left">yolov8l-seg</li><li align="left">yolov8x-seg</li></ul></details>     |    instance segmentation | Build_In |


- face alignment

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: | 
| [PIPNet](./cv/face_alignment/pipnet/README.md)  |   [official](https://github.com/jhb86253817/PIPNet/tree/master)   | <details> <summary>model name</summary><ul><li align="left">pip_resnet18</li><li align="left">pip_resnet101</li><li align="left">pipnet_mobilenet_v2</li><li align="left">pipnet_mobilenet_v3</li></ul></details> |    face alignment    |   Build_In   |
| [PFLD](./cv/face_alignment/pfld/README.md)  |  [pytorch](https://github.com/polarisZhao/PFLD-pytorch)   |  <details> <summary>model name</summary><ul><li align="left">pfld</li></ul></details>   |    face alignment    |   Build_In   |


- face detection

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: | 
| [RetinaFace](./cv/face_detection/retinaface/README.md)  | [pytorch](https://github.com/biubug6/Pytorch_Retinaface)  |    <details> <summary>model name</summary><ul><li align="left">retinaface-resnet50</li><li align="left">retinaface-mobilenet0.25</li></ul></details>    |    face detection    |   Build_In   |
| [SCRFD](./cv/face_detection/scrfd/README.md) |  [insightface](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)  |    <details> <summary>model name</summary><ul><li align="left">scrfd_500m</li><li align="left">scrfd_500m_bnkps</li><li align="left">scrfd_1g</li><li align="left">scrfd_2.5g</li><li align="left">scrfd_2.5g_bnkps</li><li align="left">scrfd_10g</li><li align="left">scrfd_10g_bnkps</li><li align="left">scrfd_34g</li></ul></details>    |    face detection    | Build_In |


- facial quality

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [SDD-FIQA](./cv/face_quality/sdd_fiqa/README.md) | [sdd_fiqa](https://github.com/Tencent/TFace/tree/quality) |    <details> <summary>model name</summary><ul><li align="left">sdd_fiqa</li></ul></details> |     face quality     |   Build_In |  


- face recognize

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: | 
| [FaceNet](./cv/face_recognize/facenet/README.md) |  [tensorflow](https://github.com/davidsandberg/facenet)   |   <details> <summary>model name</summary><ul><li align="left">facenet_vggface2</li><li align="left">facenet_casia_webface</li></ul></details>   |    face recognize    |   Build_In   |
| [FaceNet](./cv/face_recognize/facenet/README.md) |  [pytorch](https://github.com/timesler/facenet-pytorch)   |   <details> <summary>model name</summary><ul><li align="left">facenet_vggface2</li><li align="left">facenet_casia_webface</li></ul></details>   |    face recognize    |   Build_In   |


- facial attribute

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [FairFace](./cv/facial_attribute/fairface/README.md)   | [official](https://github.com/dchen236/FairFace) | <details> <summary>model name</summary><ul><li align="left">fairface_res34</li></ul></details>  |    face attribute    | Build_In |  


- image colorization

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [GPEN](./cv/image_colorization/gpen/README.md) |   [official](https://github.com/yangxy/GPEN)    | <details> <summary>model name</summary><ul><li align="left">gpen</li></ul></details> |    image colorization    | Build_In | 


- image retrieval

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [DINOv2](./cv/image_retrieval/dinov2/README.md)  |  [official](https://github.com/facebookresearch/dinov2)  |  <details> <summary>model name</summary><ul><li align="left">dinov2_vitl14_reg4</li></ul></details>  |   image_retrieval   | Build_In | 


- low light image enhancement

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [PairLIE](./cv/low_light_image_enhancement/pairlie/README.md)    |    [official](https://github.com/zhenqifu/PairLIE)    |  <details> <summary>model name</summary><ul><li align="left">pairlie</li></ul></details>  | low light image enhancement | Build_In |


- mot

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [ByteTrack](./cv/mot/bytetrack/README.md)   |   [official](https://github.com/ifzhang/ByteTrack)   |    <details> <summary>model name</summary><ul><li align="left">ByteTrack_ablation</li><li align="left">bytetrack_x_mot17</li><li align="left">bytetrack_l_mot17</li><li align="left">bytetrack_m_mot17</li><li align="left">bytetrack_s_mot17</li><li align="left">bytetrack_nano_mot17</li><li align="left">bytetrack_tiny_mot17</li><li align="left">bytetrack_x_mot20</li></ul></details>    |    mot |   Build_In   |
| [DeepSort](./cv/mot/deep_sort/README.md)   |   [pytorch](https://github.com/ZQPei/deep_sort_pytorch)   |    <details> <summary>model name</summary><ul><li align="left">fast reid</li></ul></details>    |    mot |   Build_In   |


- pose

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [OpenPose](./cv/pose/openpose/README.md)   |   [pytorch](https://github.com/Hzzone/pytorch-openpose)   |  <details> <summary>model name</summary><ul><li align="left">body model</li><li align="left">hand model</li></ul></details>   |    pose    | Build_In |
| [HRNet_Pose](./cv/pose/hrnet_pose/README.md) | [mmpose](https://github.com/open-mmlab/mmpose/blob/master/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py) |   <details> <summary>model name</summary><ul><li align="left">hrnet_pose</li></ul></details>    |    pose    | Build_In |
| [Yolov8_Pose](./cv/pose/yolov8_pose/README.md)  |  [ultralytics](https://docs.ultralytics.com/tasks/pose/)  | <details> <summary>model name</summary><ul><li align="left">yolov8n_pose</li><li align="left">yolov8s_pose</li><li align="left">yolov8m_pose</li><li align="left">yolov8l_pose</li><li align="left">yolov8x_pose</li><li align="left">yolov8x_pose_p6</li></ul></details> |    pose    | Build_In |

- reid

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
|  [Fast_Reid](./cv/reid/fast_reid/README.md)  |     [official](https://github.com/JDAI-CV/fast-reid) |    <details> <summary>model name</summary><ul><li align="left">market_bot_R50</li><li align="left">market_bot_S50</li><li align="left">market_bot_R50_ibn</li><li align="left">market_bot_R101_ibn</li></ul></details>     |    reid    |   Build_In   |


- salient object detection

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [F3Net](./cv/salient_object_detection/f3net/README.md)  |  [official](https://github.com/weijun88/F3Net)  |  <details> <summary>model name</summary><ul><li align="left">f3net</li></ul></details>  |   salient object detection   | Build_In |
| [ISNet](./cv/salient_object_detection/isnet/README.md)  |  [official](https://github.com/xuebinqin/DIS)   |  <details> <summary>model name</summary><ul><li align="left">isnet</li></ul></details>  |   salient object detection   | Build_In |


- super_resolution

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [EDSR](./cv/super_resolution/edsr/README.md) |   [basicsr](https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/docs/ModelZoo_CN.md#图像超分官方模型)    | <details> <summary>model name</summary><ul><li align="left">edsr_1x</li><li align="left">edsr_m2x</li></ul></details> |   super resolution   | Build_In |
| [EDSR](./cv/super_resolution/edsr/README.md) | [official](https://github.com/sanghyun-son/EDSR-PyTorch)  | <details> <summary>model name</summary><ul><li align="left">edsr_x2</li><li align="left">edsr_baseline_x2</li><li align="left">edsr_baseline_x4</li></ul></details> |   super resolution   | Build_In |
| [GPEN](./cv/super_resolution/gpen/README.md)  |   [official](https://github.com/yangxy/GPEN)    |  <details> <summary>model name</summary><ul><li align="left">gpen</li></ul></details>   |    face super resolution | Build_In |
|  [NCNet](./cv/super_resolution/ncnet/README.md)  |  [official](https://github.com/Algolzw/NCNet)   |  <details> <summary>model name</summary><ul><li align="left">ncnet</li></ul></details>  |   super resolution   | Build_In |
| [RCAN](./cv/super_resolution/rcan/README.md) | [official](https://github.com/yulunzhang/RCAN)  |    <details> <summary>model name</summary><ul><li align="left">rcan</li><li align="left">rcan2</li></ul></details>    |   super resolution   | Build_In |
| [RCAN](./cv/super_resolution/rcan/README.md) |     [basicsr](https://github.com/XPixelGroup/BasicSR)     |  <details> <summary>model name</summary><ul><li align="left">rcan</li></ul></details>   |   super resolution   | Build_In |
| [VDSR](./cv/super_resolution/vdsr/README.md) |    [pytorch](https://github.com/twtygqyy/pytorch-vdsr)    |  <details> <summary>model name</summary><ul><li align="left">vdsr</li></ul></details>   |   super resolution   | Build_In |


- text detection

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [DBNet](./cv/text_detection/dbnet/README.md) |   [ppocr](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/algorithm_det_db.md)    |  <details> <summary>model name</summary><ul><li align="left">dbnet_mobilenet_v3</li><li align="left">dbnet_resnet50_vd</li><li align="left">ch_PP_OCRv3_det</li><li align="left">ch_PP_OCRv4_det</li><li align="left">en_PP_OCRv3_det</li></ul></details>  |    text detection     | Build_In |
| [DBNet](./cv/text_detection/dbnet/README.md) |    [official](https://github.com/MhLiao/DB) |  <details> <summary>model name</summary><ul><li align="left">dbnet_resnet18</li><li align="left">dbnet_resnet50</li></ul></details>   |    text detection     | Build_In |
| [DBNet](./cv/text_detection/dbnet/README.md) | [mmocr](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/dbnet)  |   <details> <summary>model name</summary><ul><li align="left">dbnet_resnet18_fpnc_1200e_icdar2015</li><li align="left">dbnet_resnet18_fpnc_1200e_totaltext</li><li align="left">dbnet_resnet18_fpnc_100k_synthtext</li><li align="left">dbnet_resnet50_1200e_icdar2015</li><li align="left">dbnet_resnet50_oclip_1200e_icdar2015</li></ul></details>    |    text detection     | Build_In |


- text recognition

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [CNN_CTC](./cv/text_recognition/cnn_ctc/README.md)    |     [pytorch](https://github.com/Media-Smart/vedastr)     |    <details> <summary>model name</summary><ul><li align="left">resnet_fc</li></ul></details>    |    ocr recognize     | Build_In |
| [CRNN](./cv/text_recognition/crnn/README.md) | [ppocr](https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/doc/doc_ch/algorithm_rec_crnn.md) |   <details> <summary>model name</summary><ul><li align="left">resnet34_vd</li></ul></details>   |    text recognition     | Build_In |
| [PPOCR_V4_REC](./cv/text_recognition/ppocr_v4_rec/README.md) | [ppocr](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/PP-OCRv4_introduction.md) |   <details> <summary>model name</summary><ul><li align="left">ppocr_v4_rec</li></ul></details>   |    ocr recognize     | Build_In | 


</details>

<details><summary>NLP Models</summary>

- information extraction

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
|  [uie](./nlp/information_extraction/uie/README.md)  |   [uie_pytorch](https://github.com/HUSTAI/uie_pytorch)   |  <details> <summary>model name</summary><ul><li align="left">uie-base</li></ul></details>   | Information extraction  |  Build_In |


- named entity recognition

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [BERT](./nlp/named_entity_recognition/bert/README.md)   |  [huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) |  <details> <summary>model name</summary><ul><li align="left">bert_base_zh_ner-256</li></ul></details>  | named entity recognition |   Build_In   |
| [RoBERTa](./nlp/named_entity_recognition/roberta/README.md)     | [huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py)  |  <details> <summary>model name</summary><ul><li align="left">roberta_wwm_ext_base_zh-256</li></ul></details>  | named entity recognition |   Build_In   |


- question answering

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [BERT](./nlp/question_answering/bert/README.md)   |    [tensorflow](https://github.com/google-research/bert)    | <details> <summary>model name</summary><ul><li align="left">bert_base_en_qa-384</li><li align="left">bert_large_en_qa-384</li></ul></details> | sentence classification |   Build_In   |
| [BERT](./nlp/question_answering/bert/README.md)   |    [huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) |  <details> <summary>model name</summary><ul><li align="left">bert_base_en_qa-384</li><li align="left">bert_large_en_qa-384</li></ul></details>  | sentence classification |   Build_In   |


- sentence classification

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [BERT](./nlp/sentence_classification/bert/README.md)   |    [tensorflow](https://github.com/google-research/bert)    | <details> <summary>model name</summary><ul><li align="left">bert_base_mrpc_cls-128</li><li align="left">bert_large_mrpc_cls-128</li></ul></details> | sentence classification |   Build_In   |
| [BERT](./nlp/sentence_classification/bert/README.md)   |    [huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) |  <details> <summary>model name</summary><ul><li align="left">bert_base_mrpc_cls-128</li><li align="left">bert_base_mrpc_cls-512</li><li align="left">bert_large_mrpc_cls-128</li><li align="left">bert_large_mrpc_cls-512</li><li align="left">bert_base_imdb_cls-128</li><li align="left">bert_base_sst2_cls-128</li></ul></details>    | sentence classification |   Build_In   |
| [BERT](./nlp/sentence_classification/bert/README.md)   |  [modelscope](https://www.modelscope.cn/models/iic/nlp_structbert_sentiment-classification_chinese-base/summary)   |  <details> <summary>model name</summary><ul><li align="left">nlp_structbert_sentiment-classification_chinese-base</li></ul></details>   | sentence classification |   Build_In   |
| [Electra](./nlp/sentence_classification/electra/README.md) |  [CIB](https://drive.google.com/drive/folders/1ii0Kz6nxZujiMkoMozrWLbBCGpjmWqh2?usp=sharing)   |   <details> <summary>model name</summary><ul><li align="left">electra_small-512</li></ul></details>   | sentence classification |   Build_In   |
| [Electra](./nlp/sentence_classification/electra/README.md) |  [huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/electra/modeling_electra.py)   |   <details> <summary>model name</summary><ul><li align="left">electra_small_dc_mrpc-128</li><li align="left">electra_small_gen_mrpc-128</li><li align="left">electra_base_dc_mrpc-128</li><li align="left">electra_base_gen_mrpc-128</li><li align="left">electra_large_dc_mrpc-128</li></ul></details>    | sentence classification |   Build_In   |
| [RoBERTa](./nlp/sentence_classification/roberta/README.md) | [huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py)  |  <details> <summary>model name</summary><ul><li align="left">roberta_base_en_cls-128</li></ul></details>  | sentence classification |   Build_In   |

- Text2Vec

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [bge]((./nlp/text2vec/bge/README.md)) | [huggingface](https://huggingface.co/BAAI/) |  <details> <summary>model name</summary><ul><li align="left">bge-reranker-base</li><li align="left">bge-reranker-large</li><li align="left">bge-reranker-v2-m3</li></ul></details> | Reranker model  |  Build_In |
| [bce](./nlp/text2vec/bce/README.md) | [huggingface](https://huggingface.co/maidalun1020/bce-reranker-base_v1) |  <details> <summary>model name</summary><ul><li align="left">bce-reranker-base_v1</li></ul></details> | Reranker model  |  Build_In |



</details>


<details><summary>LLM Models</summary>

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
| [LLaMA](./llm/llama/README.md) |   [huggingface](https://huggingface.co/meta-llama)    |   <details> <summary>model name</summary><ul><li align="left">meta-llama-33b</li></ul></details>   | large language model |   Build_In   |
| [LLaMA-2](./llm/llama2/README.md) |   [huggingface](https://huggingface.co/meta-llama)    |   <details> <summary>model name</summary><ul><li align="left">Llama-2-7b-hf</li><li align="left">Llama-2-7b-chat-hf</li><li align="left">Llama-2-13b-hf</li><li align="left">Llama-2-13b-chat-hf</li><li align="left">Llama-2-70b-hf</li><li align="left">Llama-2-70b-chat-hf</li></ul></details>   | large language model |   Build_In   |
| [LLaMA-3](./llm/llama3/README.md) |   [huggingface](https://huggingface.co/meta-llama)    |   <details> <summary>model name</summary><ul><li align="left">Meta-Llama-3-8B</li><li align="left">Meta-Llama-3-8B-Instruct</li><li align="left">Meta-Llama-3-70B</li><li align="left">Meta-Llama-3-70B-Instruct</li></ul></details>   | large language model |   Build_In   |
| [LLaMA-3.1](./llm/llama3/README.md) |   [huggingface](https://huggingface.co/meta-llama)    |   <details> <summary>model name</summary><ul><li align="left">Meta-Llama-3.1-8B</li><li align="left">Meta-Llama-3.1-8B-Instruct</li><li align="left">Meta-Llama-3.1-70B</li><li align="left">Meta-Llama-3.1-70B-Instruct</li></ul></details>   | large language model |   Build_In   |
| [LLaMA-3.2](./llm/llama3/README.md) |   [huggingface](https://huggingface.co/meta-llama)    |   <details> <summary>model name</summary><ul><li align="left">Llama-3.2-1B</li><li align="left">Llama-3.2-1B-Instruct</li><li align="left">Llama-3.2-3B</li><li align="left">Llama-3.2-3B-Instruct</li></details>   | large language model |   Build_In   |
| [LLaMA-3.3](./llm/llama3/README.md) |   [huggingface](https://huggingface.co/meta-llama)    |   <details> <summary>model name</summary><ul><li align="left">Llama-3.3-70B-Instruct</li></ul></details>   | large language model |   Build_In   |
| [Internlm2](./llm/internlm2/README.md) |   [huggingface](https://huggingface.co/internlm)    |   <details> <summary>model name</summary><ul><li align="left">internlm2-1_8b</li><li align="left">internlm2-chat-1_8b</li><li align="left">internlm2-7b</li><li align="left">internlm2-chat-7b</li><li align="left">internlm2-20b</li><li align="left">internlm2-chat-20b</li></ul></details>   | large language model |   Build_In   |
| [Internlm2_5](./llm/internlm2_5/README.md) |   [huggingface](https://huggingface.co/internlm)    |   <details> <summary>model name</summary><ul><li align="left">internlm2_5-1_8b</li><li align="left">internlm2_5-1_8b-chat</li><li align="left">internlm2_5-7b</li><li align="left">internlm2_5-7b-chat</li><li align="left">internlm2_5-20b</li><li align="left">internlm2_5-20b-chat</li></ul></details>   | large language model |   Build_In   |
| [Internlm3](./llm/internlm3/README.md) |   [huggingface](https://huggingface.co/internlm)    |   <details> <summary>model name</summary><ul><li align="left">internlm3-8b-instruct</li></ul></details>   | large language model |  Build_In   |
| [ChatGLM2](./llm/chatglm2/README.md) |   [huggingface](https://huggingface.co/THUDM/chatglm2-6b)    |   <details> <summary>model name</summary><ul><li align="left">chatglm2-6b</li></ul></details>   | large language model |   Build_In   |
| [ChatGLM3](./llm/chatglm3/README.md) |   [huggingface](https://huggingface.co/THUDM/chatglm2-6b)    |   <details> <summary>model name</summary><ul><li align="left">chatglm3-6b-base</li><li align="left">chatglm3-6b</li></ul></details>   | large language model |   Build_In   |
| [ChatGLM4](./llm/chatglm4/README.md) |   [huggingface](https://huggingface.co/THUDM/chatglm2-6b)    |   <details> <summary>model name</summary><ul><li align="left">glm-4-9b-hf</li><li align="left">glm-4-9b-chat-hf</li></ul></details>   | large language model |   Build_In   | 
| [BaiChuan2](./llm/baichuan2/README.md) |   [huggingface](https://huggingface.co/baichuan-inc)    |   <details> <summary>model name</summary><ul><li align="left">Baichuan2-7B-Base</li><li align="left">Baichuan2-7B-Chat</li><li align="left">Baichuan2-13B-Base</li><li align="left">Baichuan2-13B-Chat</li></ul></details>   | large language model |   Build_In   | 
| [Qwen1.5](./llm/qwen1.5/README.md) |   [huggingface](https://huggingface.co/Qwen)    |   <details> <summary>model name</summary><ul><li align="left">Qwen1.5-0.5B</li><li align="left">Qwen1.5-0.5B-Chat</li><li align="left">Qwen1.5-1.8B</li><li align="left">Qwen1.5-1.8B-Chat</li><li align="left">Qwen1.5-4B</li><li align="left">Qwen1.5-4B-Chat</li><li align="left">Qwen1.5-7B</li><li align="left">Qwen1.5-7B-Chat</li><li align="left">Qwen1.5-14B</li><li align="left">Qwen1.5-14B-Chat</li><li align="left">Qwen1.5-32B</li><li align="left">Qwen1.5-32B-Chat</li><li align="left">Qwen1.5-72B</li><li align="left">Qwen1.5-72B-Chat</li><li align="left">Qwen1.5-110B-Chat</li></ul></details>   | large language model |   Build_In   |
| [Qwen2](./llm/qwen2/README.md) |   [huggingface](https://huggingface.co/Qwen)    |   <details> <summary>model name</summary><ul><li align="left">Qwen2-0.5B</li><li align="left">Qwen2-0.5B-Instruct</li><li align="left">Qwen2-1.5B</li><li align="left">Qwen2-1.5B-Instruct</li><li align="left">Qwen2-7B</li><li align="left">Qwen2-7B-Instruct</li><li align="left">Qwen2-72B</li><li align="left">Qwen2-72B-Instruct</li></ul></details>   | large language model |   Build_In   |
| [Qwen2.5](./llm/qwen2/README.md) |   [huggingface](https://huggingface.co/Qwen)    |   <details> <summary>model name</summary><ul><li align="left">Qwen2.5-0.5B</li><li align="left">Qwen2.5-0.5B-Instruct</li><li align="left">Qwen2.5-1.5B</li><li align="left">Qwen2.5-1.5B-Instruct</li><li align="left">Qwen2.5-3B</li><li align="left">Qwen2.5-3B-Instruct</li><li align="left">Qwen2.5-7B</li><li align="left">Qwen2.5-7B-Instruct</li><li align="left">Qwen2.5-14B</li><li align="left">Qwen2.5-14B-Instruct</li><li align="left">Qwen2.5-32B</li><li align="left">Qwen2.5-32B-Instruct</li><li align="left">Qwen2.5-72B</li><li align="left">Qwen2.5-72B-Instruct</li></ul></details>   | large language model |   Build_In   |
| [QWQ](./llm/qwq/README.md) |   [huggingface](https://huggingface.co/Qwen)    |   <details> <summary>model name</summary><ul><li align="left">QwQ-32B-Preview</li><li align="left">QwQ-32B</li></ul></details>   | large language model |   Build_In   |
| [Qwen3](./llm/qwen3/README.md) |   [huggingface](https://huggingface.co/Qwen)    |   <details> <summary>model name</summary><ul><li align="left">Qwen3-0.6B</li><li align="left">Qwen3-1.7B</li><li align="left">Qwen3-4B</li><li align="left">Qwen3-8B</li><li align= "left" > Qwen3-4B-Instruct-2507 </li><li align= "left" > Qwen3-4B-Thinking-2507 </li></ul> </details>   | large language model |   Build_In   |
| [DeepSeek-R1-Distill](./llm/deepseek_r1/README.md) |   [huggingface](https://huggingface.co/deepseek-ai)    |   <details> <summary>model name</summary><ul><li align="left">DeepSeek-R1-Distill-Qwen-1.5B</li><li align="left">DeepSeek-R1-Distill-Qwen-7B</li><li align="left">DeepSeek-R1-Distill-Qwen-14B</li><li align="left">DeepSeek-R1-Distill-Qwen-32B</li><li align="left">DeepSeek-R1-Distill-Llama-8B</li><li align="left">DeepSeek-R1-Distill-Llama-70B</li></ul></details>   | large language model |   Build_In   |




|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :-----: |
|[Qwen2.5](./llm/qwen2/README.md) | [huggingface](https://hf-mirror.com/Qwen)| <details> <summary>model name</summary><ul><li align="left">Qwen2.5-7B-Instruct-GPTQ-Int4</li><li align="left">Qwen2.5-14B-Instruct-GPTQ-Int4</li><li align="left">Qwen2.5-32B-Instruct-GPTQ-Int4</li></ul></details> | large language model | vLLM |
| [Qwen3](./llm/qwen3/README.md) |   [huggingface](https://huggingface.co/Qwen)    |   <details> <summary>model name</summary><ul><li align="left">Qwen3-30B-A3B-FP8</li><li align="left">Qwen3-30B-A3B-Instruct-2507-FP8 </li><li align= "left" > Qwen3-30B-A3B-Thinking-2507-FP8 </li></ul> </details>   | large language model |   vLLM   |
| [DeepSeek-V3](./llm/deepseek_v3/README.md) |   [huggingface](https://huggingface.co/deepseek-ai)    |   <details> <summary>model name</summary><ul><li align="left">DeepSeek-V3-Base</li><li align="left">DeepSeek-V3</li><li align="left">DeepSeek-V3-0324</li><li align="left">DeepSeek-V3.1</li><li align="left">DeepSeek-V3.1-Base</li></ul></details>   | large language model |   vLLM   |
| [DeepSeek-R1](./llm/deepseek_r1/README.md) |   [huggingface](https://huggingface.co/deepseek-ai)    |   <details> <summary>model name</summary><ul><li align="left">DeepSeek-R1</li><li align="left">DeepSeek-R1-0528</li></ul></details>   | large language model |   vLLM   |


</details>

<details><summary>VLM Models</summary>

|  model |    codebase    |  model list |    model type | runtime |
| :------: | :------: | :------: | :------: | :------: |
| [Qwen2-VL](./vlm/qwen2_vl/README.md) | [huggingface](https://hf-mirror.com/collections/Qwen/qwen2-vl-66cee7455501d7126940800d) |  <details> <summary>model name</summary><ul><li align="left">Qwen2-VL-2B-Instruct</li><li align="left">Qwen2-VL-7B-Instruct</li></ul></details> | vision-language model  |  Build_In/PyTorch |
| [Qwen2.5-VL](./vlm/qwen2_5_vl/README.md) | [huggingface](https://hf-mirror.com/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5) |  <details> <summary>model name</summary><ul><li align="left">Qwen2.5-VL-3B-Instruct</li><li align="left">Qwen2.5-VL-7B-Instruct</li><li align="left">Qwen2.5-VL-32B-Instruct</li></ul></details> | vision-language model  |  Build_In/PyTorch | 

</details>

## 免责声明
- `VastModelZOO`提供的模型仅供您用于非商业目的，请参考原始模型来源许可证进行使用
- `VastModelZOO`描述的数据集均为开源数据集，如您使用这些数据集，请参考原始数据集来源许可证进行使用
- 如您不希望您的数据集或模型公布在`VastModelZOO`上，请您提交issue，我们将尽快处理


## 使用许可
- `VastModelZOO`提供的模型，如原始模型有许可证描述，请以该许可证为准
- `VastModelZOO`遵循[Apache 2.0](LICENSE)许可证许可