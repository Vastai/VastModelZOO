
# iresnet

[Improved Residual Networks for Image and Video Recognition](https://arxiv.org/abs/2004.04989)

## Model Arch

<div align=center><img src="../../images/iresnet/iresnet.png"></div>

### pre-processing

iresnet系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作。

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

iresnet系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

该文在resnet基础上主要有三点改进
- 改进 information flow：为解决原始ResBlock块ReLU对负权值清零带来的负面影响，该文将网络分为四个Middle stage，一个Start stage和一个End stage，如上图model arch所示的C部分。stage包含ResBlock。网络根据stage的不同位置，每个ResBlock都有不同的设计。
- 改进 projection shortcut：如下图所示，对于spacial projection，该文使用stride=2的3×3max pooling层，然后，对于channel projection使用stride=1的1×1 conv，最后再跟BN。作者认为spacial projection将考虑来自特征映射的所有信息，并在下一步中选择激活度最高的元素,减少了信息的损失
<div align=center><img src="../../images/iresnet/pro_shortcut.png" width="50%" height="50%"></div>

- Grouped building block：该文在这一部分提出了分组卷积ResGroup Block来改变3×3 conv参数量较少的情况。将1×1 conv变为3×3的重分组块，如下图所示，3×3具有更多的通道和更高的空间模式学习能力。该方法比原来的ResNet多引入了4倍的空间信道
<div align=center><img src="../../images/iresnet/resgroup_block.png" width="50%" height="50%"></div>

### head

iresnet系列网络的head层由`global-average-pooling`层和一层全连接层组成

### common

- ResGroup Block
- GlobalAveragePool


## Model Info

### 模型性能

| 模型  | 源码 | top1 | top5 | flops(G) | params(M) | input size | dataset |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: | :--------: |
| iresnet50 |[official](https://github.com/iduta/iresnet)|   77.168   |   93.588   | 9.277 |    25.557    |      224    |    ImageNet    |
| iresnet101 |[official](https://github.com/iduta/iresnet)|   78.632   |   94.238   | 17.572 |    44.549    |      224    |    ImageNet    |
| iresnet152 |[official](https://github.com/iduta/iresnet)|   79.154   |   94.508   | 25.878 |    60.193    |      224    |    ImageNet    |
| iresnet200 |[official](https://github.com/iduta/iresnet)|   79.308   |   94.626   | 33.727 |    64.674    |      224    |    ImageNet    |

### 测评数据集说明

<div align=center><img src="../../images/datasets/imagenet.jpg"></div>

[ImageNet](https://image-net.org) 是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数


## Deploy

### step.1 获取模型

1. official

    ```bash
    git clone https://github.com/iduta/iresnet.git
    mv ./source_code/export.py iresnet
    cd iresnet
    python export.py --model_name iresnet50
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

1. 根据具体模型修改模型转换配置文件, 示例如下
    ```bash
    vamc build ./vacc_code/build/official_iresnet.yaml
    ```
    - [official](./vacc_code/build/official_iresnet.yaml)



### step.4 benchmark
1. 生成推理数据`npz`以及对应的`dataset.txt`
    ```bash
    python ../common/utils/image2npz.py --dataset_path /path/to/ILSVRC2012_img_val --target_path  /path/to/input_npz  --text_path npz_datalist.txt
    ```
2. 性能测试
    ```bash
    ./vamp -m iresnet50-int8-percentile-3_224_224-vacc/iresnet50 --vdsp_params ./vacc_code/vdsp_params/official-iresnet50-vdsp_params.json  -i 8 -p 1 -b 22
    ```
    
3. 获取精度信息
    ```bash
    ./vamp -m iresnet50-int8-percentile-3_224_224-vacc/iresnet50 --vdsp_params ./vacc_code/vdsp_params/official-iresnet50-vdsp_params.json  -i 8 -p 1 -b 22 --datalist npz_datalist.txt --path_output output
    ```
4. 结果解析及精度评估
    ```bash
    python ../common/eval/eval_imagenet.py --result_path output  --datalist npz_datalist.txt --label data/label/imagenet.txt
    ```
