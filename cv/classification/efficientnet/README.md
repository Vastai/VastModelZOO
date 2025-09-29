# EfficientNet

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946?context=stat.ML)
- [offical code](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

## Model Arch

<div align=center><img src="../../../images/cv/classification/efficientnet/arch.png"></div>

### pre-processing

EfficientNet系列网络的预处理操作和常见分类算法一致，`mmcls`系列在均值和方差设置上有别于常规设置，除此之外，每个子模型的输入大小均不一样

```python
[
    torchvision.transforms.Resize(),
    torchvision.transforms.CenterCrop(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
]
```

### post-processing

EfficientNet系列网络的后处理操作是对网络输出进行`softmax`作为每个类别的预测值，然后根据预测值进行排序，选择`topk`作为输入图片的预测分数以及类别

### backbone
论文中通过一个多目标的NAS来得到baseline模型（借鉴MnasNet），这里优化的目标是模型的ACC和FLOPS，其中target FLOPS是400M，最终得到了EfficientNet-B0模型，其模型架构如下表所示：
<div align=center><img src="../../../images/cv/classification/efficientnet/backbone.png"></div>
可以看到EfficientNet-B0的输入大小为224x224，首先是一个stride=2的3x3卷积层，最后是一个1x1卷积+global pooling+FC分类层，其余的stage主体是MBConv，这个指的是MobileNetV2中提出的mobile inverted bottleneck block（conv1x1-> depthwise conv3x3->conv1x1+shortcut），唯一的区别是增加了SE结构来进行优化，表中的MBConv后面的数字表示的是expand_ratio（第一个1x1卷积要扩大channels的系数）。
<div align=center><img src="../../../images/cv/classification/efficientnet/mbconv.png"></div>

### head

EfficientNet系列网络的head层由`AdaptiveAvgPool2d`层和`Linear`组成

### common

- SE
- depthwise
- ReLU/SiLU


## Model Info

### 模型性能

|        模型        |                                               源码                                                |  top1  |  top5  | FLOPs (G) | Params(M) | input size |
| :----------------: | :-----------------------------------------------------------------------------------------------: | :----: | :----: | :-------: | :-------: | :--------: |
|  efficientnet_b0   | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 77.690 | 93.530 |   0.858   |   5.29    |    224     |
|  efficientnet_b1   | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 78.796 | 94.342 |   1.266   |   7.79    |    256     |
|  efficientnet_b2   | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 80.614 | 95.316 |   2.417   |   9.11    |    288     |
|  efficientnet_b3   | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 82.240 | 96.114 |   4.359   |   12.23   |    320     |
|  efficientnet_b4   | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 83.424 | 96.596 |   9.800   |   19.34   |    384     |
| tf_efficientnet_b5 | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 83.812 | 96.748 |  22.565   |   30.39   |    456     |
| tf_efficientnet_b6 | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 84.110 | 96.888 |  41.986   |   43.04   |    528     |
| tf_efficientnet_b7 | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 84.936 | 97.204 |  83.319   |   66.35   |    600     |
| tf_efficientnet_b8 | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/efficientnet.py) | 85.368 | 97.390 |  138.325  |   87.41   |    672     |



### 测评数据集说明

<div align=center><img src="../../../images/dataset/imagenet.jpeg"></div>

ImageNet是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。`1K`代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Build_In Deploy

### step.1 获取模型
```bash
pip install timm==0.6.5
python ../common/utils/export_timm_torchvision_model.py --model_library timm  --model_name efficientnet_b0 --save_dir ./onnx  --size 224 --pretrained_weights xxx.pth
```

### step.2 获取数据集
- [校准数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../common/label/imagenet.txt)
- [label_dict](../common/label/imagenet1000_clsid_to_human.txt)

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [timm_efficientnet.yaml](./build_in/build/timm_efficientnet.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd efficientnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/timm_efficientnet.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考：[classification.py](../common/vsx/classification.py)
    ```bash
    python ../../common/vsx/classification.py \
        --file_path path/to/ILSVRC2012_img_val \
        --model_prefix_path deploy_weights/timm_efficientnet_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/timm-efficientnet_b0-vdsp_params.json \
        --label_txt path/to/imagenet.txt \
        --save_dir ./runstream_output \
        --save_result_txt mod.txt \
        --device 0
    ```

    - 精度评估
    ```
    python ../../common/eval/eval_topk.py ./runmstream_output/mod.txt
    ```

    ```
    # fp16
    top1_rate: 76.738 top5_rate: 93.094

    # int8
    top1_rate: 71.158 top5_rate: 90.2
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置[timm-efficientnet_b0-vdsp_params.json](./build_in/vdsp_params/timm-efficientnet_b0-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/timm_efficientnet_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/timm-efficientnet_b0-vdsp_params.json  -i 8 -p 1 -b 2 -s [3,224,224]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致
    
    - 数据准备，生成推理数据`npz`以及对应的`dataset.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path ILSVRC2012_img_val --target_path  input_npz  --text_path imagenet_npz.txt
    ```

    - vamp推理获取npz文件
    ```
    vamp -m deploy_weights/timm_efficientnet_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/timm-efficientnet_b0-vdsp_params.json  -i 8 -p 1 -b 22 -s [3,224,224] --datalist imagenet_npz.txt --path_output output
    ```

    - 解析输出结果用于精度评估，参考：[vamp_npz_decode.py](../common/eval/vamp_npz_decode.py)
    ```bash
    python  ../../common/eval/vamp_npz_decode.py imagenet_npz.txt output imagenet_result.txt imagenet.txt
    ```
    
    - 精度评估，参考：[eval_topk.py](../common/eval/eval_topk.py)
    ```bash
    python ../../common/eval/eval_topk.py imagenet_result.txt
    ```
