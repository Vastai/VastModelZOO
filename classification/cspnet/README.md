# CSPNet

[CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/abs/1911.11929)


## Model Arch

### pre-processing

CSPNet系列网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256/288的尺寸，然后利用`CenterCrop`算子crop出224/256的图片对其进行归一化、减均值除方差等操作

```python
crop_pct = 0.887
scale_size = int(math.floor(input_size[0] / crop_pct))
[
    torchvision.transforms.Resize(scale_size),
    torchvision.transforms.CenterCrop(input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```

### post-processing

CSPNet系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别

### backbone

CSPNet系列网络的backbone结构是由`CspStage`堆叠而成，可有效降低DenseNet和ResNet的计算量
<div align=center><img src="../../images/cspnet/block.png"></div>

### head

CSPNet系列网络的head层由`AdaptiveAvgPool2d`层和`Linear`组成

### common

- AdaptiveAvgPool2d
- Linear
- Leaky ReLU

## Model Info

### 模型性能

|     模型     |                                             源码                                              |  top1  |  top5  | flops(G) | params(M) | input size | dataset  |
| :----------: | :-------------------------------------------------------------------------------------------: | :----: | :----: | :------: | :-------: | :--------: | :------: |
| cspresnet50  |  [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/cspnet.py)  | 79.582 | 94.704 |  10.038  |  21.576   |    256     | imagenet |
| cspresnext50 |  [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/cspnet.py)  | 80.050 | 94.946 |  6.832   |  20.512   |    224     | imagenet |
| cspdarknet53 |  [timm](https://github.com/rwightman/pytorch-image-models/blob/v0.6.5/timm/models/cspnet.py)  | 80.062 | 95.084 |  14.519  |  27.607   |    256     | imagenet |
| cspresnet50  | [mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.1/configs/cspnet/README.md) | 79.55  | 94.68  |   6.96   |   21.62   |    256     | imagenet |
| cspresnext50 | [mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.1/configs/cspnet/README.md) | 79.96  | 94.96  |   6.22   |   20.57   |    224     | imagenet |
| cspdarknet53 | [mmcls](https://github.com/open-mmlab/mmclassification/blob/v0.23.1/configs/cspnet/README.md) | 80.05  | 95.07  |  10.08   |   27.64   |    256     | imagenet |
| cspdarknet53 |  [ppcls](https://github.com/PaddlePaddle/PaddleClas/blob/v2.4.0/docs/zh_CN/models/Others.md)  |  78.0  |  94.1  |  18.58   |   41.6    |    256     | imagenet |

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
    python ../common/utils/export_timm_torchvision_model.py --model_library timm  --model_name cspresnet50 --save_dir ./onnx  --size 256 --pretrained_weights xxx.pth
    ```
2. mmclassification

   mmcls框架参考 [mmclassification](https://github.com/open-mmlab/mmclassification),可使用如下位置的pytorch2onnx.py或pytorch2torchscript.py转成相应的模型
    ```bash
    git clone https://github.com/open-mmlab/mmclassification.git
    cd mmclassification

    python tools/deployment/pytorch2onnx.py \
            --config configs/cspnet/cspdarknet50_8xb32_in1k.py \
            --checkpoint weights/cspdarknet50.pth \
            --output-file output/cspdarknet50.onnx \
    ```


3. ppcls

   ```bash
    pip install PaddlePaddle==2.3.2  Paddle2ONNX==1.0.0
    paddle2onnx  --model_dir /path/to/cspnet_paddle_model/ \
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
    vamc build ./vacc_code/build/timm_cspnet.yaml
    ```
    - [timm](./vacc_code/build/timm_cspnet.yaml)
    - [mmcls](./vacc_code/build/mmcls_cspnet.yaml)
    - [ppcls](./vacc_code/build/ppcls_cspnet.yaml)


### step.4 模型推理

1. 根据step.3配置模型三件套信息，[model_info](./vacc_code/model_info/model_info_cspnet.json)
2. 配置python版数据预处理流程vdsp_params参数
   - [timm](./vacc_code/vdsp_params/sdk1.0/timm_vdsp_params_cspnet_rgb.json)
   - [mmcls](./vacc_code/vdsp_params/sdk1.0/mmcls_vdsp_params_cspnet_rgb.json)
   - [ppcls](./vacc_code/vdsp_params/sdk1.0/ppcls_vdsp_params_cspnet_rgb.json)


3. 执行推理，参考 [runstream](../common/sdk1.0/sample_cls.py)
    ```bash
    python ../common/sdk1.0/sample_cls.py --save_dir output/cspnet_result.txt
    ```

4. 精度评估
   ```bash
    python ../common/eval/eval_topk.py output/cspnet_result.txt
   ```

### step.5 benchmark

1. 生成推理数据`npz`以及对应的`datalist.txt`
    ```bash
    python ../common/utils/image2npz.py --dataset_path /path/to/ILSVRC2012_img_val --target_path  /path/to/input_npz  --text_path npz_datalist.txt
    ```
2. 性能测试
    ```bash
    ./vamp -m cspresnet50-int8-percentile-3_256_256-vacc/cspresnet50 --vdsp_params ./vacc_code/vdsp_params/vamp/timm-cspresnet50-vdsp_params.json  -i 8 -p 1 -b 16
    ```
    
3. 获取精度信息
    ```bash
    ./vamp -m cspresnet50-int8-percentile-3_256_256-vacc/cspresnet50 --vdsp_params ./vacc_code/vdsp_params/vamp/timm-cspresnet50-vdsp_params.json  -i 1 -p 1 -b 1 \
    --datalist npz_datalist.txt --path_output ./output/cspresnet50
    ```
4. 结果解析及精度评估
   ```bash
    python ../common/eval/eval_imagenet.py --result_path ./output/cspresnet50  --datalist /path/to/npz_datalist.txt --label data/label/imagenet.txt
   ```