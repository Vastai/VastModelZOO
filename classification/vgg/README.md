
# VGG

[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)


## Model Arch

<div align=center><img src="../../images/vgg/arch.png" width="50%" height="50%"></div>

### pre-processing

VGG系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

VGG系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

VGG系列网络的backbone结构可以看成是数个vgg_block的堆叠，每个vgg_block由多个conv+bn+relu或conv+relu，最好再加上池化层组成。VGG网络名称后面的数字表示整个网络中包含参数层的数量（卷积层或全连接层，不含池化层）

### head

VGG系列网络的head层为3个全连接层组成

### common

- maxpool

## Model Info

### 模型性能

|   模型   |                                                 源码                                                  |  top1  |  top5  | flops(G) | params(M) | input size |
| :------: | :---------------------------------------------------------------------------------------------------: | :----: | :----: | :------: | :-------: | :--------: |
|  vgg11   |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 69.028 | 88.626 |  7.609   |  132.863  |    224     |
| vgg11_bn |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 70.360 | 89.802 |  7.639   |  132.869  |    224     |
|  vgg13   |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 69.926 | 89.246 |  11.308  |  133.048  |    224     |
| vgg13_bn |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 71.594 | 90.376 |  11.357  |  133.054  |    224     |
|  vgg16   |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 71.590 | 90.382 |  15.47   |  138.358  |    224     |
| vgg16_bn |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 73.350 | 91.504 |  15.524  |  138.366  |    224     |
|  vgg19   |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 72.366 | 90.870 |  19.632  |  143.667  |    224     |
| vgg19_bn |       [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)        | 74.214 | 91.848 |  19.691  |  143.678  |    224     |
|  vgg11   |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 69.02  | 88.626 |  7.609   |  132.863  |    224     |
| vgg11_bn |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 70.37  | 89.81  |  7.639   |  132.869  |    224     |
|  vgg13   |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 69.928 | 89.246 |  11.308  |  133.048  |    224     |
| vgg13_bn |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 71.586 | 90.374 |  11.357  |  133.054  |    224     |
|  vgg16   |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 71.592 | 90.382 |  15.47   |  138.358  |    224     |
| vgg16_bn |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 73.36  | 91.516 |  15.524  |  138.366  |    224     |
|  vgg19   |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 72.376 | 90.876 |  19.632  |  143.667  |    224     |
| vgg19_bn |        [torchvision](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/vgg.py)         | 74.218 | 91.842 |  19.691  |  143.678  |    224     |
|  vgg11   |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11_8xb32_in1k.py)  | 68.75  | 88.87  |   7.63   |  132.86   |    224     |
| vgg11_bn | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11bn_8xb32_in1k.py) | 70.75  | 90.12  |   7.64   |  132.87   |    224     |
|  vgg13   |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg13_8xb32_in1k.py)  | 70.02  | 89.46  |  11.34   |  133.05   |    224     |
| vgg13_bn | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg13bn_8xb32_in1k.py) | 72.15  | 90.71  |  11.36   |  133.05   |    224     |
|  vgg16   |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg16_8xb32_in1k.py)  | 71.62  | 90.49  |   15.5   |  138.36   |    224     |
| vgg16_bn |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg16_8xb32_in1k.py)  | 73.72  | 91.68  |  15.53   |  138.37   |    224     |
|  vgg19   |  [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg19_8xb32_in1k.py)  | 72.41  | 90.80  |  19.67   |  143.67   |    224     |
| vgg19_bn | [mmcls](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg19bn_8xb32_in1k.py) | 74.70  | 92.24  |   19.7   |  143.68   |    224     |

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
    python ../common/utils/export_timm_torchvision_model.py --model_library timm  --model_name vgg11 --save_dir ./onnx  --size 224 --pretrained_weights xxx.pth
    ```

2. mmclassification

    mmcls框架参考 [mmclassification](https://github.com/open-mmlab/mmclassification),可使用如下位置的pytorch2onnx.py或pytorch2torchscript.py转成相应的模型
    ```bash
    git clone https://github.com/open-mmlab/mmclassification.git
    cd mmclassification

    python tools/deployment/pytorch2onnx.py \
            --config configs/resnet/vgg13_b32x8_imagenet.py \
            --checkpoint weights/vgg13.pth \
            --output-file output/vgg13.onnx \
    ```

### step.2 准备数据集
本模型使用ImageNet官网ILSVRC2012的5万张验证集进行测试，针对`int8`校准数据可从该数据集中任选1000张，为了保证量化精度，请保证每个类别都有数据，请用户自行获取该数据集，[ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/index.php)
```
├── ImageNet
|   ├── val
|   |    ├── ILSVRC2012_val_00000001.JPEG
│   |    ├── ILSVRC2012_val_00000002.JPEG
│   |    ├── ......
|   ├── val_label.txt
````

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
    vamc build ./vacc_code/build/timm_vgg.yaml
    ```
    - [timm](./vacc_code/build/timm_vgg.yaml)

### step.4 模型推理
1. 根据step.3配置模型三件套信息，[model_info](./vacc_code/model_info/model_info_vgg.json)
2. 配置数据预处理流程vdsp_params参数
   - [timm](./vacc_code/vdsp_params/sdk1.0/timm-vgg11_bn-vdsp_params.json)

3. 执行推理，参考[runstream](../common/sdk1.0/sample_cls.py)
    ```bash
    python ../common/sdk1.0/sample_cls.py --save_dir output/vgg_result.txt

4. 精度评估
   ```bash
    python ../common/eval/eval_topk.py output/vgg_result.txt
   ```

### step.5 benchmark
1. 生成推理数据`npz`以及对应的`datalist.txt`
    ```bash
    python ../common/utils/image2npz.py --dataset_path /path/to/ILSVRC2012_img_val --target_path  /path/to/input_npz  --text_path npz_datalist.txt
    ```
2. 性能测试
    ```bash
    ./vamp -m vgg11-int8-percentile-3_224_224-vacc/vgg11 --vdsp_params ./vacc_code/vdsp_params/vamp/timm-vgg11-vdsp_params.json  -i 1 -p 1 -b 1
    ```
    
3. 获取精度信息
    ```bash
    ./vamp -m vgg11-int8-percentile-3_224_224-vacc/vgg11 --vdsp_params ./vacc_code/vdsp_params/vamp/timm-vgg11-vdsp_params.json  -i 1 -p 1 -b 1 --datalist npz_datalist.txt --path_output output
    ```
4. 结果解析及精度评估
    ```bash
    python ../common/eval/eval_imagenet.py --result_path output  --datalist npz_datalist.txt --label data/label/imagenet.txt
    ```