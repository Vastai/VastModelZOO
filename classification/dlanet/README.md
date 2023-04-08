# DLANet

[Deep Layer Aggregation](https://arxiv.org/abs/1707.06484)

## Model Arch

### pre-processing

DLANet系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
crop_pct = 0.875
scale_size = int(math.floor(input_size[0] / crop_pct))
[
    torchvision.transforms.Resize(scale_size),
    torchvision.transforms.CenterCrop(input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

DLANet系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

DLANet网络中每个conv block都是resnet的残差网络，主要有两个创新点：

1. IDA  融合不同的分辨率/尺度上的feature
2. HDA 合并所有的模块和通道的feature

这两个模型类似senet可以融合到任何模块中使用。

<div align=center><img src="../../images/dlanet/dla_arch.png"></div>

### head

DLANet系列网络的head层由`AdaptiveAvgPool2d`层和`Linear`组成

### common

- Hierarchical Deep Aggregation
- Iterative Deep Aggregation
- Aggregation Node
- Downsample 2x

## Model Info

### 模型性能

|      模型      |                                           源码                                           |  top1  |  top5  | flops(G) | params(M) | input size | dataset  |
| :------------: | :--------------------------------------------------------------------------------------: | :----: | :----: | :------: | :-------: | :--------: | :------: |
|     dla34      | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) | 74.620 | 92.072 |  6.845   |  15.742   |    224     | imagenet |
|    dla46_c     | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) | 64.866 | 86.294 |  1.316   |   1.301   |    224     | imagenet |
|    dla46x_c    | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) | 65.970 | 86.980 |  1.234   |   1.068   |    224     | imagenet |
|     dla60      | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) | 77.030 | 93.320 |  9.503   |  22.037   |    224     | imagenet |
|     dla60x     | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) | 78.244 | 94.018 |  7.937   |  17.352   |    224     | imagenet |
|    dla60x_c    | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) | 67.892 | 88.426 |  1.345   |   1.320   |    224     | imagenet |
|     dla102     | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) | 78.030 | 93.948 |  16.047  |  33.269   |    224     | imagenet |
|    dla102x     | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) | 78.516 | 94.226 |  13.168  |  26.309   |    224     | imagenet |
|    dla102x2    | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) | 79.446 | 94.632 |  20.897  |  41.282   |    224     | imagenet |
|     dla169     | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) | 78.692 | 94.340 |  25.863  |  53.390   |    224     | imagenet |
| dla60_res2net  | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) | 78.462 | 94.206 |  9.272   |  20.848   |    224     | imagenet |
| dla60_res2next | [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/dla.py) | 78.440 | 94.150 |  7.804   |  17.033   |    224     | imagenet |
|     dla34      | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/DLA.md) | 76.03  | 92.98  |   6.2    |   15.8    |    224     | imagenet |
|    dla46_c     | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/DLA.md) | 63.21  | 85.30  |   1.0    |    1.3    |    224     | imagenet |
|     dla60      | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/DLA.md) | 76.10  | 92.92  |   8.4    |   22.0    |    224     | imagenet |
|     dla60x     | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/DLA.md) | 77.53  | 93.78  |   7.0    |   17.4    |    224     | imagenet |
|    dla60x_c    | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/DLA.md) | 66.45  | 87.54  |   1.2    |    1.3    |    224     | imagenet |
|     dla102     | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/DLA.md) | 78.93  | 94.52  |   14.4   |   33.3    |    224     | imagenet |
|    dla102x     | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/DLA.md) | 78.10  | 94.00  |   11.8   |   26.4    |    224     | imagenet |
|    dla102x2    | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/DLA.md) | 78.85  | 94.45  |   18.6   |   41.4    |    224     | imagenet |
|     dla169     | [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/DLA.md) | 78.09  | 94.09  |   23.2   |   53.5    |    224     | imagenet |
|     dla34      |              [ucbdrive](https://github.com/ucbdrive/dla/blob/master/dla.py)              | 74.42  | 92.06  |  6.875   |  15.784   |    224     | imagenet |
|    dla46_c     |              [ucbdrive](https://github.com/ucbdrive/dla/blob/master/dla.py)              | 64.27  | 86.004 |  1.320   |   1.310   |    224     | imagenet |
|    dla46x_c    |              [ucbdrive](https://github.com/ucbdrive/dla/blob/master/dla.py)              | 65.576 | 86.948 |  1.238   |   1.077   |    224     | imagenet |
|    dla60x_c    |              [ucbdrive](https://github.com/ucbdrive/dla/blob/master/dla.py)              | 67.894 | 88.316 |  1.353   |   1.337   |    224     | imagenet |
|     dla60      |              [ucbdrive](https://github.com/ucbdrive/dla/blob/master/dla.py)              | 76.844 | 93.23  |  9.678   |  22.334   |    224     | imagenet |
|     dla60x     |              [ucbdrive](https://github.com/ucbdrive/dla/blob/master/dla.py)              | 78.150 | 94.102 |  8.112   |  17.650   |    224     | imagenet |
|     dla102     |              [ucbdrive](https://github.com/ucbdrive/dla/blob/master/dla.py)              | 77.894 | 93.968 |  16.339  |  33.732   |    224     | imagenet |
|    dla102x     |              [ucbdrive](https://github.com/ucbdrive/dla/blob/master/dla.py)              | 78.604 | 94.166 |  13.460  |  26.772   |    224     | imagenet |
|    dla102x2    |              [ucbdrive](https://github.com/ucbdrive/dla/blob/master/dla.py)              | 79.406 | 94.538 |  21.189  |  41.745   |    224     | imagenet |
|     dla169     |              [ucbdrive](https://github.com/ucbdrive/dla/blob/master/dla.py)              | 78.542 | 94.214 |  26.245  |  53.989   |    224     | imagenet |


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
    python ../common/utils/export_timm_torchvision_model.py --model_library timm  --model_name dla34 --save_dir ./onnx  --size 224 --pretrained_weights xxx.pth
    ```

2. ucbdrive
    ```bash
    cd ./source_code
    git clone https://github.com/ucbdrive/dla.git
    mv export_onnx.py dla & cd dla
    python export_onnx.py
    ```

3. ppcls
   ```bash
    pip install PaddlePaddle==2.3.2  Paddle2ONNX==1.0.0

    paddle2onnx  --model_dir /path/to/dlanet_paddle_model/ \
                --model_filename model.pdmodel \
                --params_filename model.pdiparams \
                --save_file model.onnx \
                --enable_dev_version False \
                --opset_version 10
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

1. 使用模型转换工具vamc，根据具体模型修改模型转换配置文件, 此处以`timm` 为例

   ```bash
   vamc build ./vacc_code/build/timm_dlanet.yaml
   ```
   - [timm](./vacc_code/build/timm_dlanet.yaml)
   - [ucbdrive](./vacc_code/build/timm_dlanet.yaml)
   - [ppcls](./vacc_code/build/ppcls_dlanet.yaml)




### step.4 benchmark

1. 生成推理数据`npz`以及对应的`datalist.txt`
    ```bash
    python ../common/utils/image2npz.py --dataset_path /path/to/ILSVRC2012_img_val --target_path  /path/to/input_npz  --text_path npz_datalist.txt
    ```
2. 性能测试
    ```bash
    ./vamp -m dla34-int8-kl_divergence-3_224_224-vacc/dla34 --vdsp_params ./vacc_code/vdsp_params/timm-dla34-vdsp_params.json  -i 8 -p 1 -b 48  
    ```
    
3. 获取精度信息
    ```bash
    ./vamp -m dla34-int8-kl_divergence-3_224_224-vacc/dla34 --vdsp_params ./vacc_code/vdsp_params/timm-dla34-vdsp_params.json  -i 8 -p 1 -b 48  --datalist npz_datalist.txt --path_output output
    ```
4. 结果解析及精度评估
   ```bash
    python ../common/eval/eval_imagenet.py --result_path output  --datalist npz_datalist.txt --label data/label/imagenet.txt
   ```