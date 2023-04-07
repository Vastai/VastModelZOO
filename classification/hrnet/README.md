
# HRNet

[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/pdf/1908.07919.pdf)


## Model Arch

<div align=center><img src="../../images/hrnet/cls-hrnet.png"></div>

### pre-processing

HRNet系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

HRNet系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

HRNet网络是将不同分辨率的feature map进行并联，在并联的基础上添加不同分辨率feature map之间的融合，具体融合的方法可以分为4种：

1. 同分辨率的层直接复制
2. 需要升分辨率的使用bilinear upsample + 1x1卷积将channel数统一
3. 需要降分辨率的使用stride为2的3x3 卷积
4. 三个feature map融合的方式是相加

通过上述规则生成了一系列特征层的组合，然后选择相应的特征组合，即组成了基于HRNet算法的分类网络

### head

HRNet系列网络的head层由global-average-pooling层和一层全连接层组成

### common

- bilinear upsample

## Model Info

### 模型性能

|           模型           |                                                源码                                                 |  top1  |  top5  | flops(G) | params(M) | input size |
| :----------------------: | :-------------------------------------------------------------------------------------------------: | :----: | :----: | :------: | :-------: | :--------: |
| HRNet_w18_small_model_v1 |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  72.3  |  90.7  |   1.49   |   13.2    |    224     |
| HRNet_w18_small_model_v2 |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  75.1  |  92.4  |   2.42   |   15.6    |    224     |
|        HRNet_w18         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  76.8  |  93.3  |   3.99   |   21.3    |    224     |
|        HRNet_w30         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  78.2  |  94.2  |   7.55   |   37.7    |    224     |
|        HRNet_w32         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  78.5  |  94.2  |   8.31   |   41.2    |    224     |
|        HRNet_w40         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  78.9  |  94.5  |   11.8   |   57.6    |    224     |
|        HRNet_w44         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  78.9  |  94.4  |   13.9   |   67.1    |    224     |
|        HRNet_w48         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  79.3  |  94.5  |   16.1   |   77.5    |    224     |
|        HRNet_w64         |                   [official](https://github.com/HRNet/HRNet-Image-Classification)                   |  79.5  |  94.6  |   26.9   |   128.1   |    224     |
|        HRNet_w18         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 76.75  | 93.44  |   4.33   |   21.30   |    224     |
|        HRNet_w30         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 78.19  | 94.22  |   8.17   |   37.71   |    224     |
|        HRNet_w32         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 78.44  | 94.19  |   8.99   |   41.23   |    224     |
|        HRNet_w40         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 78.94  | 94.47  |  12.77   |   57.55   |    224     |
|        HRNet_w44         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 78.88  | 94.37  |  14.96   |   67.06   |    224     |
|        HRNet_w48         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 79.32  | 94.52  |  17.36   |   77.47   |    224     |
|        HRNet_w64         | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/hrnet.py) | 79.46  | 94.65  |  29.00   |  128.06   |    224     |
|    hrnet_w18_small_v1    |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 72.336 | 90.68  |  3.611   |  13.187   |    224     |
|    hrnet_w18_small_v2    |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 75.11  | 92.416 |  5.856   |  15.597   |    224     |
|        hrnet_w18         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 76.76  | 93.444 |  9.667   |  21.299   |    224     |
|        hrnet_w30         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 78.198 | 94.224 |  18.207  |  37.712   |    224     |
|        hrnet_w32         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 78.452 | 94.188 |  20.026  |  41.233   |    224     |
|        hrnet_w40         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 78.922 | 94.47  |  28.436  |  57.557   |    224     |
|        hrnet_w44         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 78.896 | 94.37  |  33.320  |  67.065   |    224     |
|        hrnet_w48         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      |  79.3  | 94.514 |  38.658  |  77.470   |    224     |
|        hrnet_w64         |     [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/hrnet.py)      | 79.47  | 94.654 |  64.535  |  128.060  |    224     |

### 测评数据集说明

<div align=center><img src="../../images/datasets/imagenet.jpg"></div>

[ImageNet](https://image-net.org) 是一个计算机视觉系统识别项目，是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。能够从图片中识别物体。ImageNet是一个非常有前景的研究项目，未来用在机器人身上，就可以直接辨认物品和人了。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万张图像中，还提供了边界框。ImageNet包含2万多个类别; 一个典型的类别，如“气球”或“草莓”，每个类包含数百张图像。

ImageNet数据是CV领域非常出名的数据集，ISLVRC竞赛使用的数据集是轻量版的ImageNet数据集。ISLVRC2012是非常出名的一个数据集，在很多CV领域的论文，都会使用这个数据集对自己的模型进行测试，在该项目中分类算法用到的测评数据集就是ISLVRC2012数据集的验证集。在一些论文中，也会称这个数据叫成ImageNet 1K或者ISLVRC2012，两者是一样的。“1 K”代表的是1000个类别。

### 评价指标说明

- top1准确率: 测试图片中最佳得分所对应的标签是正确标注类别的样本数除以总的样本数
- top5准确率: 测试图片中正确标签包含在前五个分类概率中的个数除以总的样本数

## Deploy
📝 注：该网络仅在`step.1 & step.3`部分有区别

### step.1 获取模型

1. timm

    ```bash
    pip install timm==0.6.5
    python ../common/utils/export_timm_torchvision_model.py --model_library timm  --model_name hrnet_w30 --save_dir ./onnx  --size 224 --pretrained_weights xxx.pth
    ```

2. mmclassification

    mmcls框架参考 [mmclassification](https://github.com/open-mmlab/mmclassification),可使用如下位置的pytorch2onnx.py或pytorch2torchscript.py转成相应的模型

    ```bash
    git clone https://github.com/open-mmlab/mmclassification.git
    cd mmclassification

    python tools/deployment/pytorch2onnx.py \
            --config configs/hrnet/hrnet-w32_4xb32_in1k.py \
            --checkpoint weights/hrnet_w32.pth \
            --output-file output/hrnet_w32.onnx \
    ```

3. official

   进入hrnet子文件夹，该项目可以实现模型转换至torchscript与onnx格式，转换时可以指定模型路径以及相应的模型配置文件，运行命令如下：

    ```bash
    git clone https://github.com/HRNet/HRNet-Image-Classification.git
    mv source_code/export.py HRNet-Image-Classification
    cd HRNet-Image-Classification

    python tools/export.py --cfg_file experiments/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --weight_path /path/to/weights_path --save_name hrnetv2_w64
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
    vamc build ./vacc_code/build/timm_hrnet.yaml
    ```
    - [timm](./vacc_code/build/timm_hrnet.yaml)
    - [official](./vacc_code/build/official_hrnet.yaml)
    - [mmcls](./vacc_code/build/mmcls_hrnet.yaml)

### step.4 模型推理
1. 根据step.3配置模型三件套信息，[model_info](./vacc_code/model_info/model_info_hrnet.json)
2. 配置python版数据预处理流程vdsp_params参数
   - [vdsp_params](./vacc_code/vdsp_params/sdk1.0/timm-hrnet-vdsp_params.json)


3. 执行推理，参考[runstream](../common/sdk1.0/sample_cls.py)
    ```bash
    python ../common/sdk1.0/sample_cls.py --save_dir output/hrnet_result.txt
    ```

4. 精度评估
   ```bash
    python ../common/eval/eval_topk.py output/hrnet_result.txt
   ```


### step.5 benchmark
1. 生成推理数据`npz`以及对应的`datalist.txt`
    ```bash
   python ../common/utils/image2npz.py --dataset_path /path/to/ILSVRC2012_img_val --target_path  /path/to/input_npz  --text_path npz_datalist.txt
    ```
2. 性能测试
    ```bash
    ./vamp -m hrnet_w18-int8-percentile-3_256_256-vacc/hrnet_w18 --vdsp_params ./vacc_code/vdsp_params/vamp/timm-hrnet_w18-vdsp_params.json  -i 1 -p 1 -b 1
    ```
    
3. 获取精度信息
    ```bash
    ./vamp -m hrnet_w18-int8-percentile-3_256_256-vacc/hrnet_w18 --vdsp_params ./vacc_code/vdsp_params/vamp/timm-hrnet_w18-vdsp_params.json  -i 1 -p 1 -b 1 --datalist npz_datalist.txt --path_output output
    ```
4. 结果解析及精度评估
    ```bash
    python ../common/eval/eval_imagenet.py --result_path output  --datalist npz_datalist.txt --label data/label/imagenet.txt
    ```
## appending

注意：`mmcls`转换为onnx时，op_set需设置为11，10会报错。
