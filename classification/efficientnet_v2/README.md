# EfficientNet_V2

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)


## Model Arch

### pre-processing

EfficientNetV2系列网络的预处理操作和常见分类算法一致，主要区别是每个子模型的输入大小均不一样

```python
[
    torchvision.transforms.Resize(),
    torchvision.transforms.CenterCrop(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
]
```

### post-processing

EfficientNetV2系列网络的后处理操作是对网络输出进行`softmax`作为每个类别的预测值，然后根据预测值进行排序，选择`topk`作为输入图片的预测分数以及类别

### backbone

EfficientNetV2中除了使用到MBConv模块外，还使用了Fused-MBConv模块（主要是在网络浅层中使用，将浅层MBConv结构替换成Fused-MBConv结构能够明显提升训练速度），Fused-MBConv结构非常简单，即将原来的MBConv结构（之前在将EfficientNetv1时有详细讲过）主分支中的expansion conv1x1和depthwise conv3x3替换成一个普通的conv3x3，如下图所示

<div align=center><img src="../../images/efficientnetv2/fused-mbconv.png"></div>

论文基于trainning-aware NAS framework，搜索工作主要还是基于之前的Mnasnet以及EfficientNet. 但是这次的优化目标联合了accuracy、parameter efficiency以及trainning efficiency三个维度。下表为efficientnetv2_s的模型架构

<div align=center><img src="../../images/efficientnetv2/arch.png"></div>

另外，论文提出了改进的渐进学习方法，该方法会根据训练图像的尺寸动态调节正则方法(例如dropout、data augmentation和mixup)。通过实验展示了该方法不仅能够提升训练速度，同时还能提升准确率。

### head

EfficientNetV2系列网络的head层由`AdaptiveAvgPool2d`层和`Linear`组成

### common

- SE
- depthwise
- MBConv
- Fused-MBConv


## Model Info

### 模型性能

|        模型         |                                               源码                                                |  top1  |  top5  | FLOPs (G) | Params(M) | input size |
| :-----------------: | :-----------------------------------------------------------------------------------------------: | :----: | :----: | :-------: | :-------: | :--------: |
| efficientnetv2_rw_t | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 82.344 | 96.196 |   7.032   |  13.556   |    288     |
| efficientnetv2_rw_s | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 83.810 | 96.724 |  19.201   |  23.780   |    384     |
| efficientnetv2_rw_m | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 84.812 | 97.146 |  47.406   |  52.929   |    416     |



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
    pip install timm==0.6.5
    python ../common/utils/export_timm_torchvision_model.py --model_library timm  --model_name efficientnetv2_rw_t --save_dir ./onnx  --size 224 --pretrained_weights xxx.pth
    ```

### step.2 获取数据集
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

1. 使用模型转换工具vamc，根据具体模型修改模型转换配置文件
    ```bash
   vamc build ./vacc_code/build/timm_efficientnet_v2.yaml
   ```


### step.4 benchmark

1. 生成推理数据`npz`以及对应的`dataset.txt`
    ```bash
    python ../common/utils/image2npz.py --dataset_path /path/to/ILSVRC2012_img_val --target_path  /path/to/input_npz  --text_path npz_datalist.txt
    ```
2. 性能测试
    ```bash
    ./vamp -m efficientnet_v2-int8-percentile-3_256_256-vacc/efficientnet_v2 --vdsp_params ./vacc_code/vdsp_params/timm-efficientnet_v2-vdsp_params.json  -i 16 -p 1 -b 20
    ```
    
3. 获取精度信息
    ```bash
    ./vamp -m efficientnet_v2-int8-kl_divergence-3_224_224-vacc/efficientnet_v2 --vdsp_params ./vacc_code/vdsp_params/timm-efficientnet_v2-vdsp_params.json  -i 16 -p 1 -b 20  --datalist npz_datalist.txt --path_output output
    ```
4. 结果解析及精度评估
   ```bash
   python ../common/eval/vamp_eval.py --result_path output  --datalist npz_datalist.txt --label data/label/imagenet.txt
   ```
