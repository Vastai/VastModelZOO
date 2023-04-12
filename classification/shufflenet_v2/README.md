
# ShuffleNetV2

[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

## Model Arch

<div align=center><img src="../../images/shufflenetv2/arch.png" width="50%" height="50%"></div>

### pre-processing

ShuffleNetV2系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

ShuffleNetV2系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

ShuffleNetV2系列网络利用channel split将输入通道切分为两个分支，其中一个分支保持不变，另一个分支包含三个恒等通道数的卷积层，之后再将两个分组concat为一个，再执行通道随机化操作

<div align=center><img src="../../images/shufflenetv2/s-block.png" width="50%" height="50%"></div>

### head

ShuffleNetV2系列网络的head层由global-average-pooling层和一层全连接层组成

### common

- channel split
- channel shuffle
- depthwise conv
- group conv

## Model Info

### 模型性能

| 模型  | 源码 | top1 | top5 | flops(M) | params(M) | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| shufflenet_v2_x0.5 |[torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/shufflenetv2.py)|  60.552   |   81.746   |   44.572    |    1.367    |        224    |
| shufflenet_v2_x1.0 |[torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/shufflenetv2.py)|  69.362   |   88.316   |   152.709    |    2.279    |        224    |
| shufflenet_v2_x1.0 |[mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.1/mmcls/models/backbones/shufflenet_v2.py)|  69.550   |   88.920   |   149.000    |    2.280    |        224    |
| shufflenet_v2_x0.25 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  49.900   |   73.790   |   18.950    |    0.610    |        224    |
| shufflenet_v2_x0.33 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  53.730   |   77.050   |   24.040    |    0.650    |        224    |
| shufflenet_v2_x0.5 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  60.320   |   82.260   |   42.580    |    1.370    |        224    |
| shufflenet_v2_x1.0 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  68.800   |   88.450   |   148.860    |    2.290    |        224    |
| shufflenet_v2_x1.5 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  71.630   |   90.150   |   301.35    |    3.530    |        224    |
| shufflenet_v2_x2.0 |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  73.150   |   91.200   |   571.700    |    7.400    |        224    |
| shufflenet_v2_swish |[ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/ppcls/arch/backbone/model_zoo/shufflenet_v2.py)|  70.030   |   89.170   |   148.860    |    2.290    |        224    |
| shufflenet_v2_x0.5 |[megvii](https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/README.md)| 61.1   |   82.6   |   41    |    1.4    |        224    |
| shufflenet_v2_x1.0 |[megvii](https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/README.md)|  69.4   |   88.9   |   146    |    2.3    |        224    |
| shufflenet_v2_x1.5 |[megvii](https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/README.md)|  72.6   |   90.6  |   299    |    3.5    |        224    |
| shufflenet_v2_x2.0 |[megvii](https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/README.md)|  75   |   92.4   |   591    |    7.4    |        224    |

### 测评数据集说明

<div align=center><img src="../../images/datasets/imagenet.jpg"></div>

[ImageNet](https://image-net.org) 是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Deploy

### step.1 获取模型

1. torchvision

    ```bash
    python ../common/utils/export_timm_torchvision_model.py --model_library torchvision  --model_name shufflenet_v2 --save_dir ./onnx  --size 224 --pretrained_weights xxx.pth
    ```

2. ppcls

    ```bash
    pip install PaddlePaddle==2.3.2  Paddle2ONNX==1.0.0

    paddle2onnx  --model_dir /path/to/resnet_paddle_model/ \
                --model_filename model.pdmodel \
                --params_filename model.pdiparams \
                --save_file model.onnx \
                --enable_dev_version False \
                --opset_version 10
    ```
3. megvii
    ```bash
    git clone https://github.com/megvii-model/ShuffleNet-Series.git
    mv source_code/export_onnx.py ShuffleNet-Series/ShuffleNetV2
    cd ShuffleNet-Series/ShuffleNetV2
    python export_onnx.py --modelsize 0.5x
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

1. 使用模型转换工具vamc，根据具体模型修改配置文件,此处以`megvii` 为例
    > 注意megvii的输入为BGR形式
    ```bash
    vamc build ./vacc_code/build/megvii_shufflenetv2.yaml
    ```
    - [megvii](./vacc_code/build/megvii_shufflenetv2.yaml)
    - [ppcls](./vacc_code/build/ppcls_shufflenetv2.yaml)
    - [mmcls](./vacc_code/build/mmcls_shufflenetv2.yaml)
    - [torchvision](./vacc_code/build/torchvision_shufflenetv2.yaml)




### step.4 benchmark
1. 生成推理数据`npz`以及对应的`datalist.txt`
    ```bash
    python ../common/utils/image2npz.py --dataset_path /path/to/ILSVRC2012_img_val --target_path  /path/to/input_npz  --text_path npz_datalist.txt
    ```
2. 性能测试
    ```bash
    ./vamp -m shufflenet_v2-int8-percentile-3_299_299-vacc/shufflenet_v2 --vdsp_params ./vacc_code/vdsp_params/timm-shufflenet_v2-vdsp_params.json  -i 1 -p 1 -b 1
    ```
    
3. 获取精度信息
    ```bash
    ./vamp -m shufflenet_v2-int8-percentile-3_299_299-vacc/shufflenet_v2 --vdsp_params ./vacc_code/vdsp_params/timm-shufflenet_v2-vdsp_params.json  -i 1 -p 1 -b 1 --datalist npz_datalist.txt --path_output output
    ```
4. 结果解析及精度评估
   ```bash
   python ../common/eval/vamp_eval.py --result_path output  --datalist npz_datalist.txt --label data/label/imagenet.txt
   ```