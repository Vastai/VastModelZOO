## official

```bash
git clone https://github.com/dog-qiuqiu/FastestDet.git
cd FastestDet
git checkout 50473cd155cb088aa4a99e64ff6a4b3c24fa07e1
```

### step.1 获取预训练模型

官方提供onnx/torchscript转换脚本，直接运行`test.py`脚本即可，运行前，需要修改`module/custom_layers.py`中的代码，如下

```python
class DetectHead(nn.Module):
   def __init__(self, input_channels, category_num):
      super(DetectHead, self).__init__()
      self.conv1x1 =  Conv1x1(input_channels, input_channels)

      self.obj_layers = Head(input_channels, 1)
      self.reg_layers = Head(input_channels, 4)
      self.cls_layers = Head(input_channels, category_num)

      #self.sigmoid = nn.Sigmoid()
      #self.softmax = nn.Softmax(dim=1)
      
   def forward(self, x):
      x = self.conv1x1(x)
      
      obj = self.obj_layers(x)
      reg = self.reg_layers(x)
      cls = self.cls_layers(x)
      return obj, reg, cls

      # return torch.cat((obj, reg, cls), dim =1)

```

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [official_fastestdet.yaml](../build_in/build/official_fastestdet.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd fastestdet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_fastestdet.yaml
    ```

### step.4 模型推理

- 参考[fastestdet_vsx.py](../build_in/vsx/python/fastestdet_vsx.py)生成预测的txt结果

    ```
    python ../build_in/vsx/python/fastestdet_vsx.py \
        --file_path /path/to/coco_val2017 \
        --model_prefix_path deploy_weights/official_fastestdet_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-fastestdet-vdsp_params.json \
        --label_txt /path/to/coco.txt \
        --save_dir ./infer_output \
        --device 0
    ```

- 参考[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt ./infer_output
    ```

    <details><summary>点击查看精度统计结果</summary>

    ```
    # fp16
    DONE (t=1.79s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.128
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.250
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.119
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.022
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.125
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.220
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.139
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.197
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.201
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.032
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.198
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.355
    {'bbox_mAP': 0.128, 'bbox_mAP_50': 0.25, 'bbox_mAP_75': 0.119, 'bbox_mAP_s': 0.022, 'bbox_mAP_m': 0.125, 'bbox_mAP_l': 0.22, 'bbox_mAP_copypaste': '0.128 0.250 0.119 0.022 0.125 0.220'}

    # int8
    DONE (t=1.97s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.116
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.234
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.105
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.018
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.110
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.205
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.130
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.185
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.189
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.026
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.183
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.343
    {'bbox_mAP': 0.116, 'bbox_mAP_50': 0.234, 'bbox_mAP_75': 0.105, 'bbox_mAP_s': 0.018, 'bbox_mAP_m': 0.11, 'bbox_mAP_l': 0.205, 'bbox_mAP_copypaste': '0.116 0.234 0.105 0.018 0.110 0.205'}
    ```

    </details>

### step.5 性能精度测试
1. 性能测试
    ```bash
    vamp -m deploy_weights/official_fastestdet_int8/mod --vdsp_params ../build_in/vdsp_params/official-fastestdet-vdsp_params.json -i 2 p 2 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```

    - vamp推理，获得npz结果
    ```bash
    vamp -m deploy_weights/official_fastestdet_int8/mod --vdsp_params ../build_in/vdsp_params/official-fastestdet-vdsp_params.json -i 2 p 2 -b 1 --datalist npz_datalist.txt --path_output npz_output
    ```

    - npz文件解析，[npz_decode.py](../build_in/vdsp_params/npz_decode.py)
    ```bash
    python ../build_in/vdsp_params/npz_decode.py --txt result_npz --label_txt path/to/coco.txt --input_image_dir path/to/coco_val2017 --model_size 352 352 --vamp_datalist_path datalist_npz.txt --vamp_output_dir npz_output
    ```

    - 精度计算， [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
    ```bash
        python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt result_npz
    ```
