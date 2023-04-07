# RepVGG

- [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
- [offical code](https://github.com/DingXiaoH/RepVGG)

## Model Arch

### pre-processing

RepVGG系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

RepVGG系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

RepVGG系列网络的backbone结构是由`RepVGGBlock`堆叠而成，用结构重参数化（structural re-parameterization）实现VGG式单路极简架构，一路3x3卷到底，在速度和性能上达到SOTA水平，在ImageNet上超过80%正确率
<div align=center><img src="../../images/repvgg/block.png"></div>

同时，也可以选择是否加载[SEBlock](https://arxiv.org/abs/1709.01507), 其结构如下：
<div align=center><img src="../../images/repvgg/se.png"></div>


### head

CSPNet系列网络的head层由`AdaptiveAvgPool2d`层和`Linear`组成

### common

- AdaptiveAvgPool2d
- Linear
- all 3x3 Conv
- SEBlock
- ReLU


### deploy
- 训练完成后，我们对模型做等价转换，得到部署模型。这一转换也非常简单，因为1x1卷积是一个特殊（卷积核中有很多0）的3x3卷积，而恒等映射是一个特殊（以单位矩阵为卷积核）的1x1卷积！根据卷积的线性（具体来说是可加性），每个RepVGG Block的三个分支可以合并为一个3x3卷积；
- 假设输入和输出通道都是2，故3x3卷积的参数是4个3x3矩阵，1x1卷积的参数是一个2x2矩阵。注意三个分支都有BN（batch normalization）层，其参数包括累积得到的均值及标准差和学得的缩放因子及bias。这并不会妨碍转换的可行性，因为推理时的卷积层和其后的BN层可以等价转换为一个带bias的卷积层（也就是通常所谓的**吸BN**）
- 对三分支分别**吸BN**之后（注意恒等映射可以看成一个“卷积层”，其参数是一个2x2单位矩阵！），将得到的1x1卷积核用0给pad成3x3。最后，三分支得到的卷积核和bias分别相加即可。这样，每个RepVGG Block转换前后的输出完全相同，因而训练好的模型可以等价转换为只有3x3卷积的单路模型
<div align=center><img src="../../images/repvgg/deploy.png"></div>


## Model Info

### 模型性能

| 模型  | 源码 | top1  | top5 | FLOPs (B) | Params(M) | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| RepVGG-A0 |[official](https://github.com/DingXiaoH/RepVGG) <br>[mmcls](https://github.com/open-mmlab/mmclassification/tree/v0.23.2/configs/repvgg)</br>|   72.41   |   90.50   | 1.4 |    8.30    |    224    |
|  RepVGG-A1  | [official](https://github.com/DingXiaoH/RepVGG)<br/>[mmcls](https://github.com/open-mmlab/mmclassification/tree/v0.23.2/configs/repvgg)</br> | 74.47 | 91.85 |    2.4    |   12.78   |    224     |
|  RepVGG-A2  | [official](https://github.com/DingXiaoH/RepVGG)<br>[mmcls](https://github.com/open-mmlab/mmclassification/tree/v0.23.2/configs/repvgg)</br> | 76.48 | 93.01 |    5.1    |   25.49   |    224     |
|  RepVGG-B0  | [official](https://github.com/DingXiaoH/RepVGG)<br>[mmcls](https://github.com/open-mmlab/mmclassification/tree/v0.23.2/configs/repvgg)</br> | 75.14 | 92.42 |    3.1    |   14.33   |    224     |
|  RepVGG-B1  | [official](https://github.com/DingXiaoH/RepVGG)<br/>[mmcls](https://github.com/open-mmlab/mmclassification/tree/v0.23.2/configs/repvgg)</br> | 78.37 | 94.11 |   11.8    |   51.82   |    224     |
| RepVGG-B1g2 | [official](https://github.com/DingXiaoH/RepVGG)<br>[mmcls](https://github.com/open-mmlab/mmclassification/tree/v0.23.2/configs/repvgg)</br> | 77.79 | 93.88 |   8.82    |   41.36   |    224     |
| RepVGG-B1g4 | [official](https://github.com/DingXiaoH/RepVGG)<br/>[mmcls](https://github.com/open-mmlab/mmclassification/tree/v0.23.2/configs/repvgg)</br> | 77.58 | 93.84 |   7.32    |   36.13   |    224     |
|  RepVGG-B2  | [official](https://github.com/DingXiaoH/RepVGG)<br/>[mmcls](https://github.com/open-mmlab/mmclassification/tree/v0.23.2/configs/repvgg)</br> | 78.78 | 94.42 |   18.39   |   80.32   |    224     |
| RepVGG-B2g4 | [official](https://github.com/DingXiaoH/RepVGG)<br/>[mmcls](https://github.com/open-mmlab/mmclassification/tree/v0.23.2/configs/repvgg)</br> | 79.38 | 94.68 |   11.34   |   55.78   |    224     |
|  RepVGG-B3  | [official](https://github.com/DingXiaoH/RepVGG)<br/>[mmcls](https://github.com/open-mmlab/mmclassification/tree/v0.23.2/configs/repvgg)</br> | 80.52 | 95.26 |   26.22   |  110.96   |    224     |
| RepVGG-B3g4 | [official](https://github.com/DingXiaoH/RepVGG)<br/>[mmcls](https://github.com/open-mmlab/mmclassification/tree/v0.23.2/configs/repvgg)</br> | 80.22 | 95.10 |   16.08   |   75.63   |    224     |



### 测评数据集说明

<div align=center><img src="../../images/datasets/imagenet.jpg"></div>

[ImageNet](https://image-net.org) 是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。`1K`代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Deploy
### step.1 获取模型
- 预训练模型导出参考[README](./source_code/README.md)

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

1. 使用模型转换工具vamc，根据具体模型修改模型转换配置文件, 以下为示例
    ```bash
    vamc build ./vacc_code/build/official_repvgg.yaml
    ```
    - [official](./vacc_code/build/official_repvgg.yaml)


### step.4 模型推理

1. 根据step.3配置模型三件套信息，[model_info](./vacc_code/model_info/model_info_repvgg.json)
2. 配置python版数据预处理流程vdsp_params参数
   - [official](./vacc_code/vdsp_params/sdk1.0/official-repvgg_a0-vdsp_params.json)
3. 执行推理，参考[runstream](../common/sdk1.0/sample_cls.py)
    ```bash
    python ../common/sdk1.0/sample_cls.py --save_dir output/repvgg_result.txt
    ```

4. 精度评估
   ```bash
    python ../common/eval/eval_topk.py output/repvgg_result.txt
   ```

### step.5 benchmark
1. 生成推理数据`npz`以及对应的`datalist.txt`
    ```bash
    python ../common/utils/image2npz.py --dataset_path /path/to/ILSVRC2012_img_val --target_path  /path/to/input_npz  --text_path npz_datalist.txt
    ```
2. 性能测试
    ```bash
    ./vamp -m ./vacc/RepVGG_B1g4-fp16-none-3_224_224-vacc/RepVGG_B1g4 --vdsp_params ./vacc_code/vdsp_params/official-repvgg_b1-vdsp_params.json  -i 1 -p 1 -b 1
    ```
    

3. 获取精度信息
    ```bash
    ./vamp -m ./vacc/RepVGG_B1g4-fp16-none-3_224_224-vacc/RepVGG_B1g4 --vdsp_params ./vacc_code/vdsp_params/official-repvgg_b1-vdsp_params.json  -i 1 -p 1 -b 1 --datalist npz_datalist.txt --path_output output
    ```
4. 结果解析及精度评估
   ```bash
    python ../common/eval/eval_imagenet.py --result_path output  --datalist npz_datalist.txt --label data/label/imagenet.txt
    ```
