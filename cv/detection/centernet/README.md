<div  align="center">
<img src="../../../images/cv/detection/centernet/info.png" width="80%" height="80%">
</div>

# CenterNet

[Objects as Points](https://arxiv.org/abs/1904.07850)

## Code Source
```
# official
link: https://github.com/xingyizhou/CenterNet
branch: master
commit: 4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c

# mmdet
link: https://github.com/open-mmlab/mmdetection
branch: v2.25.0
commit: ca11860f4f3c3ca2ce8340e2686eeaec05b29111
```

## Model Arch
![](../../../images/cv/detection/centernet/arch.png)

### pre-processing

`centernet`系列的预处理主要是对输入图片仿射变换后进行归一化操作并减均值除方差，然后送入网络forward即可，均值方差的设置如下

```python
mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)
```

### post-processing

`centernet`系列的后处理操作是利用网络预测特征图进行box decode，然后进行nms操作

### backbone

论文中CenterNet提到了三种用于目标检测的网络，这三种网络都是编码解码(encoder-decoder)的结构：

- Resnet-18 with up-convolutional layers : 28.1% coco and 142 FPS
- DLA-34 : 37.4% COCOAP and 52 FPS
- Hourglass-104 : 45.1% COCOAP and 1.4 FPS


### head

backbone每个网络内部的结构不同，但是在模型的最后都是加了三个网络构造来输出预测值，默认是80个类、2个预测的中心点坐标、2个中心点的偏置。

用官方的源码(使用Pytorch)来表示一下最后三层，其中hm为heatmap、wh为对应中心点的width和height、reg为偏置量

```
(hm): Sequential(
(0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace)
(2): Conv2d(64, 80, kernel_size=(1, 1), stride=(1, 1))
)
(wh): Sequential(
(0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace)
(2): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
)
(reg): Sequential(
(0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace)
(2): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
)
```

### common
- warpAffine
- residual layer


## Model Info

### 模型性能

|    模型    |                       源码                       | mAP@.5:.95 | mAP@.5 | flops(G) | params(M) | input size |
| :--------: | :----------------------------------------------: | :--------: | :----: | :------: | :-------: | :--------: |
|   centernet_res18   | [official](https://github.com/xingyizhou/CenterNet) |    25.6    |  43.5  | 90.016  |  15.820   |    512     |
|   centernet_res18 **vacc fp16** | - |    25.6    |  43.5  | -  |  -   |    512     |
|   centernet_res18 **vacc int8 kl_divergence** | - |    25.1    |  43.0  | -  |  -  |    512     |
|   centernet_res18 **vacc fp16** | - |    25.0    |  41.0  | -  |  -   |    512     |
|   centernet_res18 **vacc int8 kl_divergence** | - |    24.6    |  40.6  | -  |  -  |    512     |


### 测评数据集说明

![](../../../images/dataset/coco.png)

[MS COCO](https://cocodataset.org/#download)数据集，MS COCO的全称是Microsoft Common Objects in Context，是微软于2014年出资标注的Microsoft COCO数据集，与ImageNet竞赛一样，被视为是计算机视觉领域最受关注和最权威的比赛数据集之一。

COCO数据集支持目标检测、关键点检测、实例分割、全景分割与图像字幕任务。在图像检测任务中，COCO数据集提供了80个类别，验证集包含5000张图片，上表的结果即在该验证集下测试。

### 评价指标说明

- mAP: mean of Average Precision, 检测任务评价指标，多类别的AP的平均值；AP即平均精度，是Precision-Recall曲线下的面积
- mAP@.5: 即将IoU设为0.5时，计算每一类的所有图片的AP，然后所有类别求平均，即mAP
- mAP@.5:.95: 表示在不同IoU阈值（从0.5到0.95，步长0.05）上的平均mAP

## Build_In Deploy

#### step.1 获取预训练模型

```bash
git clone https://github.com/xingyizhou/CenterNet.git
cd CenterNet
git checkout 4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c
```

官方未提供onnx转换脚本，可以在项目'src/lib/detectors/ctdet.py'文件30行插入如下代码
```python
input_names = ["input"]
output_names = ["output"]
inputs = torch.randn(1, 3, self.opt.input_h, self.opt.input_w)

torch_out = torch.onnx._export(self.model, inputs, 'centernet.onnx', export_params=True, verbose=False,
                            input_names=input_names, output_names=output_names, opset_version=10)
```

然后，运行项目demo即可

```bash
cd src

python demo.py ctdet --demo ../images/16004479832_a748d55f21_k.jpg --load_model ../models/model_best.pth --arch res_18 --gpus -1 --fix_res
```

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../common/label/coco.txt)

### step.3 模型转换
1. 根据具体模型修改配置文件
    - [official_centernet.yaml](./build_in/build/official_centernet.yaml)

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`

2. 模型编译
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd centernet
    mkdir workspace
    cd workspace
    vamc compile ./build_in/build/official_centernet.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights/official_centernet_run_stream_int8`文件夹，其中包含转换后的模型文件。

### step.4 模型推理
1. 由于该测试依赖centernet的官方代码，因此需要先安装官方代码依赖，如下：
```
cd source_code
git clone https://github.com/xingyizhou/CenterNet.git
cd CenterNet
git checkout 4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c

cd src/lib/external
make
```

2. 参考[vsx脚本](./build_in/vsx/python/centernet_vsx.py)，修改参数并运行如下脚本
    ```bash
    python ./build_in/vsx/python/centernet_vsx.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/official_centernet_run_stream_int8/mod \
        --vdsp_params_info ./build_in/vdsp_params/official-centernet_res18-vdsp_params.json \
        --label_txt path/to/coco.txt \
        --save_dir ./runstream_output \
        --device 0
    ```
    - 注意替换命令行中--file_path和gt_path为实际路径

3. [eval_map.py](../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt ./runstream_output
   ```
   - 测试精度信息如下
   ```
   DONE (t=6.19s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.212
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.363
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.215
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.053
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.219
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.361
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.215
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.335
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.350
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.121
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.373
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.584
   ```

### step.5 性能精度测试
1. 性能测试，修改vdsp参数[official-centernet_res18-vdsp_params.json](./build_in/vdsp_params/official-centernet_res18-vdsp_params.json)：
    ```bash
    vamp -m deploy_weights/official_centernet_run_stream_int8/mod \
    --vdsp_params ./build_in/vdsp_params/official-centernet_res18-vdsp_params.json \
    -i 2 p 2 -b 1 -s [1,512,512]
    ```

2. 精度测试
    - 数据准备，基于[image2npz.py](../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`datalist_npz.txt`
    ```bash
    python ../common/utils/image2npz.py \
        --dataset_path path/to/coco_val2017 \
        --target_path  path/to/coco_val2017_npz  \
        --text_path datalist_npz.txt
    ```
    - vamp推理，获取npz结果输出
    ```bash
    vamp -m deploy_weights/official_centernet_run_stream_int8/mod \
    --vdsp_params ./build_in/vdsp_params/official-centernet_res18-vdsp_params.json \
    -i 2 p 2 -b 1 -s [1,512,512] \
    --datalist datalist_npz.txt \
    --path_output outputs/centernet
    ```

3. 基于[vamp_decode.py](./build_in/vdsp_params/vamp_decode.py)，解析vamp输出的npz文件，进行绘图和保存txt结果：
    ```bash
    python ./build_in/vdsp_params/vamp_decode.py \
    --gt_dir datasets/coco_val2017 \
    --input_npz_path datalist_npz.txt \
    --out_npz_dir outputs/centernet \
    --draw_dir coco_val2017_npz_result
    ```

4. 基于[eval_map.py](../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标：
   ```bash
    python ../common/eval/eval_map.py \
    --gt path/to/instances_val2017.json \
    --txt path/to/vamp_draw_output
   ```
