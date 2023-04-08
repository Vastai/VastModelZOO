# SENet

- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

## Model Arch

### pre-processing

SENet系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

SENet系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

Squeeze-and-Excitation Networks（SENet）是由自动驾驶公司Momenta在2017年公布的一种全新的图像识别结构，它通过对特征通道间的相关性进行建模，把重要的特征进行强化来提升准确率。

下图是SENet的Block单元，其中X和U是普通卷积层的输入（C'xH'xW'）和输出（CxHxW），这些都是以往结构中已存在的。SENet增加的部分是U后的结构：对U先做一个Global Average Pooling（作者称为Squeeze过程），输出的1x1xC数据再经过两级全连接（作者称为Excitation过程），最后用sigmoid（论文中的self-gating mechanism）限制到[0，1]的范围，把这个值作为scale乘到U的C个通道上， 作为下一级的输入数据。这种结构的原理是想通过控制scale的大小，把重要的特征增强，不重要的特征减弱，从而让提取的特征指向性更强。

<div align=center><img src="../../images/senet/seblock.png"></div>

下图是两个SENet实际应用的例子，左侧是SE-Inception的结构，即Inception模块和SENet组和在一起；右侧是SE-ResNet，ResNet和SENet的组合，这种结构scale放到了直连相加之前。

<center class="half">
    <img src="../../images/senet/se-inception_module.png" width="50%"/><img src="../../images/senet/se-resnet_module.png" width="50%"/>
</center>

### head

SENet系列网络的head层由`AdaptiveAvgPool2d`层和`Linear`组成

### common

- Global Average Pooling
- 1x1 Conv
- sigmoid


## Model Info

### 模型性能

| 模型  | 源码 | top1  | top5 | FLOPs (G) | Params(M) | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| seresnet50 |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)| 80.264 | 95.072| 9.192 |  28.088   |    224    |
| seresnet152d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)| 84.360 | 97.040| 53.738 |  66.841   |    320    |
| seresnext26d_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)| 	77.604 | 93.608| 6.118 |  16.810   |    224    |
| seresnext26t_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)| 77.976 | 93.746| 6.046 |  16.807   |    224    |
| seresnext50_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)| 81.258 | 95.630| 9.535 |  27.560   |    224    |
| seresnext101_32x8d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)| 84.204 | 96.876| 60.763 |  93.569   |    288    |
| seresnext101d_32x8d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/resnet.py)| 84.362 | 96.918| 61.656 |  93.588   |    288    |
| legacy_senet154 |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/senet.py)| 81.310 | 95.496| 46.318 |  115.089   |    224    |
| legacy_seresnet18 |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/senet.py)| 71.742 | 90.332| 4.054 |  11.779   |    224    |
| legacy_seresnet34 |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/senet.py)| 74.808 | 92.126| 8.175 |  21.959   |    224    |
| legacy_seresnet50 |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/senet.py)| 77.630 | 93.750| 8.673 |  28.088   |    224    |
| legacy_seresnet101 |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/senet.py)| 78.388 | 94.264| 16.973 |  49.327   |    224    |
| legacy_seresnet152 |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/senet.py)| 78.652 | 	94.370| 25.283 |  66.822   |    224    |
| legacy_seresnext26_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/senet.py)| 77.106 | 93.318| 5.577 |  16.790   |    224    |
| legacy_seresnext50_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/senet.py)| 79.068 | 94.434| 9.535 |  27.560   |    224    |
| legacy_seresnext101_32x4d |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/senet.py)| 80.224 | 95.010| 17.911 |  48.955   |    224    |
| seresnet50 |[mmcls](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)| 77.74 | 93.84| 8.26 |  28.09   |    224    |
| seresnet101 |[mmcls](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)| 78.26 | 94.07| 15.72 |  49.33   |    224    |
| seresnet18 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)| 73.3 | 91.4| 4.14 |  11.8   |    224    |
| seresnet34 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)| 76.5 | 93.2| 7.84 |  21.98   |    224    |
| seresnet50 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)| 79.5 | 94.8| 8.67 |  28.09   |    224    |
| senet154 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)| 81.4 | 95.5| 45.83 |  114.29   |    224    |
| seresnext50_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)| 78.4 | 94| 8.02 |  26.16   |    224    |
| seresnext50_32x4d_vd |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)| 80.2 | 94.9| 10.76 |  26.28   |    224    |
| seresnext101_32x4d |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/SEResNext_and_Res2Net.md)| 79.39 | 94.43| 15.02 |  46.28   |    224    |


### 测评数据集说明

<div align=center><img src="../../images/datasets/imagenet.jpg"></div>

[ImageNet](https://image-net.org) 是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。`1K`代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Deploy
### step.1 获取模型

1. timm
    ```bash
    python cls_mode_hub.py \
            --model_library timm \
            --model_name seresnet50 \
            --save_dir output/ \
            --pretrained_weights weights/seresnet50.pth \
            --convert_mode pt \
    ```
2. mmclassification

    mmcls框架参考 [mmclassification](https://github.com/open-mmlab/mmclassification),可使用如下位置的pytorch2onnx.py或pytorch2torchscript.py转成相应的模型
    ```bash
    git clone https://github.com/open-mmlab/mmclassification.git
    cd mmclassification

    python tools/deployment/pytorch2onnx.py \
            --config configs/resnext/seresnet50_b32x8_imagenet.py \
            --checkpoint weights/seresnet50.pth \
            --output-file output/seresnet50.onnx \
    ```

3. paddle
    ```bash
    pip install PaddlePaddle==2.3.2  Paddle2ONNX==1.0.0

    paddle2onnx  --model_dir /path/to/senet_paddle_model/ \
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
   - [mmcls](./vacc_code/build/mmcls_senet.yaml)
   - [ppcls](./vacc_code/build/ppcls_senet.yaml)
   - [timm](./vacc_code/build/timm_senet.yaml)

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
    ./vamp -m seresnet50-int8-max-3_224_224-vacc/seresnet50 --vdsp_params ./vacc_code/vdsp_params/timm-seresnet50-vdsp_params.json  -i 8 -p 1 -b 22
    ```
    
3. 获取精度信息
    ```bash
    ./vamp -m seresnet50-int8-max-3_224_224-vacc/seresnet50 --vdsp_params ./vacc_code/vdsp_params/timm-seresnet50-vdsp_params.json  -i 8 -p 1 -b 22 --datalist npz_datalist.txt --path_output output
    ```
4. 结果解析及精度评估
   ```bash
   python ../common/eval/eval_imagenet.py --result_path output  --datalist npz_datalist.txt --label data/label/imagenet.txt
   ```
