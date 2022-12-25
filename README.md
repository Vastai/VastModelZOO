![logo](./images/logo.png)

# VastModelZOO

为方便大家使用VastAI modelzoo，我们将持续增加典型网络和基础插件。如果您有任何需求，请提交issues，我们会及时处理。


## model tools

## model summary

按算法方向主要从以下两个方面总结：

- arch summary : 该算法方向的整体发展脉络和重点模型
- op&module summary：该算法方向的算子清单和通用结构


## model list

|                             model                              |                                                       codebase                                                       |                                                                                                                                                                                                                                                                                                                                                        model list                                                                                                                                                                                                                                                                                                                                                         |      model type       | runtime |
| :------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------: | :-----: |
|          [ResNet](./classification/resnet/README.md)           |             [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)              |                                                                                                                                                                                                                                     <details> <summary>model name</summary><ul><li align="left">resnet18</li><li align="left">resnet26</li><li align="left">resnet34</li><li align="left">resnet50</li><li align="left">resnet101</li><li align="left">resnet152</li><li align="left">gluon_resnet18_v1b</li></ul></details>                                                                                                                                                                                                                                      |    classification     |   E2E   |
|          [ResNet](./classification/resnet/README.md)           |              [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/resnet.py)               |                                                                                                                                                                                                                                                    <details> <summary>model name</summary><ul><li align="left">resnet18</li><li align="left">resnet34</li><li align="left">resnet50</li><li align="left">resnet101</li><li align="left">resnet152</li></ul></details>                                                                                                                                                                                                                                                     |    classification     |   E2E   |
|          [ResNet](./classification/resnet/README.md)           |            [mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.2/configs/resnet/README.md)             |                                                                                                                                                                                                                                                    <details> <summary>model name</summary><ul><li align="left">resnet18</li><li align="left">resnet34</li><li align="left">resnet50</li><li align="left">resnet101</li><li align="left">resnet152</li></ul></details>                                                                                                                                                                                                                                                     |    classification     |   E2E   |
|          [ResNet](./classification/resnet/README.md)           |             [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/ResNet.md)              |                                                                      <details> <summary>model name</summary><ul><li align="left">resnet18</li><li align="left">resnet18_vd</li><li align="left">resnet34</li><li align="left">resnet34_vd</li><li align="left">resnet34_vd_ssld</li><li align="left">resnet50</li><li align="left">resnet50_vc</li><li align="left">resnet50_vd</li><li align="left">resnet50_vd_ssld</li><li align="left">resnet101</li><li align="left">resnet101_vd</li><li align="left">resnet101_vd_ssld</li><li align="left">resnet152</li><li align="left">resnet152_vd</li><li align="left">resnet200_vd</li></ul></details>                                                                      |    classification     |   E2E   |
|          [ResNet](./classification/resnet/README.md)           |             [keras](https://github.com/keras-team/keras/blob/2.3.1/keras/applications/resnet.py)              |                                                                      <details> <summary>model name</summary><ul><li align="left">resnet50</li><li align="left">resnet50v2</li><li align="left">resnet101</li><li align="left">resnet101v2</li><li align="left">resnet152</li><li align="left">resnet152v2</li></ul></details>                                                                      |    classification     |   E2E   |