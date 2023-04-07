<div align=center><img src="../../images/ghostnet/flops.png"></div>

# GhostNet

[GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)


## Model Arch

<div align=center><img src="../../images/ghostnet/ghost.png"></div>

### pre-processing

GhostNet系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

GhostNet系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

论文通过对比分析ResNet-50网络第一个残差组（Residual group）输出的特征图可视化结果，发现一些特征图高度相似。如果按照传统的思考方式，可能认为这些相似的特征图存在冗余，是多余信息，想办法避免产生这些高度相似的特征图。

但是论文推测CNN的强大特征提取能力和这些相似的特征图（Ghost对）正相关，不去刻意的避免产生这种Ghost对，而是尝试利用简单的线性操作来获得更多的Ghost对。因此论文提出了Ghost Module（分为常规卷积、Ghost生成和特征图拼接三步），如下图

<div align=center><img src="../../images/ghostnet/ghost-module.jpg"></div>

Ghost Module和深度分离卷积很类似，不同之处在于先进行PointwiseConv，后进行DepthwiseConv，另外增加了DepthwiseConv的数量，包括一个恒定映射。

Ghost BottleNeck整体架构和Residual Block非常相似，也可以直接认为是将Residual Block中的卷积操作用Ghost Module（GM）替换得到，如下图：

<div align=center><img src="../../images/ghostnet/ghost-block.png"></div>

### head

GhostNet系列网络的head层由global-average-pooling层和一层全连接层组成

### common

- DepthwiseConv
- Ghost Module

## Model Info

### 模型性能

|     模型     |                                             源码                                              |  top1  |  top5  | flops(G) | params(M) | input size |
| :----------: | :-------------------------------------------------------------------------------------------: | :----: | :----: | :------: | :-------: | :--------: |
| ghostnet_100 | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/ghostnet.py) | 73.974 | 91.460 |  0.342   |    3.9    |    224     |


### 测评数据集说明

<div align=center><img src="../../images/datasets/imagenet.jpg"></div>

[ImageNet](https://image-net.org) 是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数


## Deploy
### step.1 获取模型

```bash
pip install timm==0.6.5
python ../common/utils/export_timm_torchvision_model.py --model_library timm  --model_name ghostnet_100 --save_dir ./onnx  --size 224 --pretrained_weights xxx.pth
```

### step.2 获取数据集
- 本模型使用ImageNet官网ILSVRC2012的5万张验证集进行测试，针对`int8`校准数据可从该数据集> 中任选1000张，为了保证量化精度，请保证每个类别都有数据，请用户自行获取该数据集，[ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/index.php)

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
1. 根据具体模型修改模型转换配置文件

   ```bash
   vamc build ./vacc_code/build/timm_ghostnet.yaml
   ```

### step.4 模型推理
1. 根据step.3配置模型三件套信息，[model_info](./vacc_code/model_info/model_info_ghostnet.json)
2. 配置python版数据预处理流程vdsp_params参数
   - [timm](./vacc_code/vdsp_params/sdk1.0/timm-ghostnet-vdsp_params.json)


3. 执行推理，参考[runstream](../common/sdk1.0/sample_cls.py)
    ```bash
    python ../common/sdk1.0/sample_cls.py --save_dir output/ghostnet_result.txt
    ```

4. 精度评估
   ```bash
    python ../common/eval/eval_topk.py output/ghostnet_result.txt
   ```

### step.5 benchmark

1. 生成推理数据`npz`以及对应的`datalist.txt`
    ```bash
    python ../common/utils/image2npz.py --dataset_path /path/to/ILSVRC2012_img_val --target_path  /path/to/input_npz  --text_path npz_datalist.txt
    ```
2. 性能测试
    ```bash
    ./vamp -m ghostnet-int8-percentile-3_224_224vacc/ghostnet --vdsp_params ./vacc_code/vdsp_params/vamp/timm-ghostnet-vdsp_params.json  -i 1 -p 1 -b 1
    ```
    
3. 获取精度信息
    ```bash
    ./vamp -m ghostnet_100-int8-kl_divergence-3_224_224-vacc/ghostnet_100 --vdsp_params ./vacc_code/vdsp_params/vamp/timm-ghostnet-vdsp_params.json  -i 1 -p 1 -b 1  --datalist npz_datalist.txt --path_output output
    ```
4. 结果解析及精度评估
   ```bash
    python ../common/eval/eval_imagenet.py --result_path output  --datalist npz_datalist.txt --label data/label/imagenet.txt
    ```