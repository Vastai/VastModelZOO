
# ResNeXt

- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

## Model Arch

<div align=center><img src="../../images/resnext/arch.png" width="50%" height="50%"></div>

### pre-processing

ResNeXt系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

ResNeXt系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

ResNeXt提出了一种介于普通卷积和深度可分离卷积的这种策略：分组卷积，通过控制分组的数量（基数）来达到两种策略的平衡。分组卷积的思想是源自Inception，不同于Inception的需要人工设计每个分支，ResNeXt的每个分支的拓扑结构是相同的。最后再结合残差网络，得到的便是最终的ResNeXt。

<div align=center><img src="../../images/resnext/block.png"></div>

### head

ResNeXt系列网络的head层由global-average-pooling层和一层全连接层组成

### common

- group convolution

## Model Info

### 模型性能

| 模型  | 源码 | top1 | top5 | MACs(G) | params(M) | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| resnext50_32x4d |[torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/resnet.py)|   77.618   |  93.698   |   4.288    |    25.029    |        224    |
| resnext101_32x8d |[torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/resnet.py)   |   79.312   |   94.526  | 16.539      |  88.791      |      224     |
| resnext50_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)   |   81.108  |   95.326  | 4.288      |  25.029      |      224      |
| resnext50d_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)   |   79.670   |   94.864  | 4.531      |  25.048      |      224      |
| resnext101_32x8d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)    | 79.316  |   94.518  | 16.539     |  88.791      |      224      |
| resnext101_64x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)  |   83.140  |   96.370  | 15.585      | 83.455       |      224      |
| tv_resnext50_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)|   77.616   |   93.700   |   4.288    |   25.029     |     224       |
| gluon_resnext50_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/gluon_resnet.py)  |   79.364  |   94.426  | 4.765      | 25.029       |      224      |
| gluon_resnext101_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/gluon_resnet.py)  |   80.344  |   94.926  | 8.95      | 44.178       |      224      |
| gluon_resnext101_64x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/gluon_resnet.py)  |   80.604  |   94.992  | 17.317      | 83.455       |      224      |
| resnext50_32x4d |[mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.2/configs/resnext/README.md)|   77.90   |   93.66   |   4.27    |   25.03     |     224       |
| resnext101_32x4d |[mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.2/configs/resnext/README.md)|   78.61   |   94.17   |   8.03    |   44.18     |     224       |
| resnext101_32x8d |[mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.2/configs/resnext/README.md)|   79.27   |   94.58   |   16.5    |   88.79     |     224       |
| resnext152_32x4d |[mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.2/configs/resnext/README.md)|   78.88   |   94.33   |   11.8    |   59.95    |     224       |


| 模型  | 源码 | top1 | top5 | FLOPS(G) | params(M) | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| resnext50_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   77.8   |   93.8   |   8.02    |   23.64    |     224       |
| resnext50_64x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   78.4   |   94.1   |   15.06    |   42.36    |     224       |
| resnext50_vd_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   79.6   |   94.6   |   8.5    |   23.66    |     224       |
| resnext50_vd_64x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   80.1   |   94.9   |   15.54    |   42.38    |     224       |
| resnext101_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   78.7   |   94.2   |   15.01    |   41.54    |     224       |
| resnext101_64x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   78.4   |   94.5   |   29.05    |   78.12    |     224       |
| resnext101_vd_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   80.3   |   95.1   |   15.49    |   41.56    |     224       |
| resnext101_vd_64x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   80.8   |   95.2   |   29.53    |   78.14    |     224       |
| resnext152_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   79.0   |   94.3   |   22.01    |  56.28   |     224       |
| resnext152_64x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   79.5   |   94.7   |   43.03    |   107.57    |     224       |
| resnext152_vd_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   80.7   |   95.2   |   22.49    |  56.3   |     224       |
| resnext152_vd_64x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)|   81.1   |   95.3   |   43.52    |   107.59    |     224       |
| resnext101_32x8d_wsl |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/EfficientNet_and_ResNeXt101_wsl.md)|   82.6   |   96.7   |   29.14    |   78.44    |     224       |
| resnext101_32x16d_wsl |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/EfficientNet_and_ResNeXt101_wsl.md)|   84.2   |   97.3   |  57.55    |   152.66    |     224       |
| resnext101_32x32d_wsl |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/EfficientNet_and_ResNeXt101_wsl.md)|  85.0   |   97.6   |   115.17    |  303.11    |     224       |

**Note:** `fix_resnext101_32x48d_wsl`模型与`resnext101_32x48d_wsl`模型在转换onnx格式时失败

### 测评数据集说明

<div align=center><img src="../../images/datasets/imagenet.jpg"></div>

[ImageNet](https://image-net.org) 是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Deploy
### step.1 获取模型

1. timm、torchvision
    ```bash
    python cls_mode_hub.py \
            --model_library timm \
            --model_name resnext50_32x4d \
            --save_dir output/ \
            --pretrained_weights weights/resnext50_32x4d.pth \
            --convert_mode pt \
    ```
2. mmclassification

    mmcls框架参考 [mmclassification](https://github.com/open-mmlab/mmclassification),可使用如下位置的pytorch2onnx.py或pytorch2torchscript.py转成相应的模型
    ```bash
    git clone https://github.com/open-mmlab/mmclassification.git
    cd mmclassification
    python tools/deployment/pytorch2onnx.py \
            --config configs/resnext/resnext50_32x4d_b32x8_imagenet.py \
            --checkpoint weights/resnext50_32x4d.pth \
            --output-file output/resnext50_32x4d.onnx \
    ```

3. paddle
    ```bash
    pip install PaddlePaddle==2.3.2  Paddle2ONNX==1.0.0

    paddle2onnx  --model_dir /path/to/resnext_paddle_model/ \
                --model_filename model.pdmodel \
                --params_filename model.pdiparams \
                --save_file model.onnx \
                --enable_dev_version False \
                --opset_version 10
    ```

### step.2 准备数据集
- 本模型使用ImageNet官网ILSVRC2012的5万张验证集进行测试，针对`int8`校准数据可从该数据集中任选1000张，为了保证量化精度，请保证每个类别都有数据，请用户自行获取该数据集，[ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/index.php)

    ```
    ├── ImageNet
    |   ├── val
    |   |    ├── ILSVRC2012_val_00000001.JPEG
    │   |    ├── ILSVRC2012_val_00000002.JPEG
    │   |    ├── ......
    |   ├── val_label.txt
    ```

    ```bash
    sh ./data_prep_sh_files/valprep.sh
    ```

    ```bash
    # label.txt
    tench, Tinca tinca
    goldfish, Carassius auratus
    ...
    ```

### step.3 模型转换
1. 使用模型转换工具vamc，根据具体模型修改配置文件
   - [torchvision](./vacc_code/build/torchvision_resnext.yaml)
   - [mmcls](./vacc_code/build/mmcls_resnext.yaml)
   - [ppcls](./vacc_code/build/ppcls_resnext.yaml)
   - [timm](./vacc_code/build/timm_renext.yaml)

2. 命令行执行转换

   ```bash
   vamc build ./vacc_code/build/xxx.yaml
   ```


### step.4 benchmark
1. 生成推理数据`npz`以及对应的`datalist.txt`
    ```bash
    python ../common/utils/image2npz.py --dataset_path /path/to/ILSVRC2012_img_val --target_path  /path/to/input_npz  --text_path npz_datalist.txt
    ```
2. 性能测试
    ```bash
    ./vamp -m resnext101_32x4d-int8-max-3_224_224-vacc/resnext101_32x4d --vdsp_params ./vacc_code/vdsp_params/timm-resnext101_32x4d-vdsp_params.json  -i 8 -p 1 -b 22
    ```
    
3. 获取精度信息
    ```bash
    ./vamp -m resnext101_32x4d-int8-max-3_224_224-vacc/resnext101_32x4d --vdsp_params ./vacc_code/vdsp_params/timm-resnext101_32x4d-vdsp_params.json  -i 8 -p 1 -b 22 --datalist npz_datalist.txt --path_output output
    ```
4. 结果解析及精度评估
    ```bash
    python ../common/eval/vamp_eval.py --result_path output  --datalist npz_datalist.txt --label data/label/imagenet.txt
    ```
