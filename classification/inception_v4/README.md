
# inception_v4

[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)


## Model Arch

<div align=center><img src="../../images/inceptionv4/arch.png"></div>

### pre-processing

inception_v4系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至342的尺寸，然后利用`CenterCrop`算子crop出299的图片对其进行归一化、减均值除方差等操作。需要注意的是，inception_v3系列所用到的均值方差与其他resnet、vgg等网络所用的均值方差有所不同

```python
[
    torchvision.transforms.Resize(342),
    torchvision.transforms.CenterCrop(299),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],),
]
```

### post-processing

inception_v4系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

这个文章的最大贡献是把Residual Connection用在了InceptionNet中，并通过实验说明Residual Connection的作用仅是加快了训练速度

- stem结构：用于对进入Inception模块前的数据进行预处理。stem部分其实就是多次卷积＋２次pooling，pooling采用了Inception-v3论文里提到的卷积＋pooling并行的结构，来防止bottleneck问题
<div align=center><img src="../../images/inceptionv4/stem.png"></div>


- Inception Ａ、 Inception Ｂ、 Inception Ｃ
<div align=center><img src="../../images/inceptionv4/inception.png"></div>

- Reduction
<div align=center><img src="../../images/inceptionv4/reduction.png"></div>

### head

inception_v4系列网络的head层由`global-average-pooling`层和一层全连接层组成

### common

- ReLU
- AvgPool2d


## Model Info

### 模型性能

| 模型  | 源码 | top1 | top5 | flops(G) | params(M) | input size | dataset |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: | :--------: |
| inception v4 |[timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/inception_v4.py)|   80.162   |   94.966   | 27.375 |    42.680    |      299    |    ImageNet    |

### 测评数据集说明

<div align=center><img src="../../images/datasets/imagenet.jpg"></div>

[ImageNet](https://image-net.org) 是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Deploy

### step.1 获取模型

1. timm

    ```bash
    pip install timm==0.6.5
    python ../common/utils/export_timm_torchvision_model.py --model_library timm  --model_name inception_v4 --save_dir ./onnx  --size 299 --pretrained_weights xxx.pth
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

1. 使用模型转换工具vamc
    ```bash
    vamc build ./vacc_code/build/timm_inceptionv4.yaml
    ```
    - [timm](./vacc_code/build/timm_inceptionv4.yaml)


### step.4 benchmark
1. 生成推理数据`npz`以及对应的`datalist.txt`
    ```bash
    python ../common/utils/image2npz.py --dataset_path /path/to/ILSVRC2012_img_val --target_path  /path/to/input_npz  --text_path npz_datalist.txt
    ```
2. 性能测试
    ```bash
    ./vamp -m inception_v4-int8-percentile-3_299_299-vacc/inception_v4 --vdsp_params ./vacc_code/vdsp_params/timm-inception_v4-vdsp_params.json  -i 8 -p 1 -b 22
    ```
    
3. 获取精度信息
    ```bash
    ./vamp -m inception_v4-int8-percentile-3_299_299-vacc/inception_v4 --vdsp_params ./vacc_code/vdsp_params/timm-inception_v4-vdsp_params.json  -i 8 -p 1 -b 22 --datalist npz_datalist.txt --path_output output
    ```
4. 结果解析及精度评估
    ```bash
    python ../common/eval/vamp_eval.py --result_path output  --datalist npz_datalist.txt --label data/label/imagenet.txt
    ```