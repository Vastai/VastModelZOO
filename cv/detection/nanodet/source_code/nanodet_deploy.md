### step.1 获取预训练模型

目前仅支持不带后处理的模型进行三件套转换，因此在转换onnx以及torchscript格式前需修改下代码，主要修改部分是`/PATH/to/nanodet/nanodet/model/head/nanodet_plus_head.py`中`_forward_onnx`函数以及`forward`函数去掉后处理部分，如下：

```python
def _forward_onnx(self, feats):
    """only used for onnx export"""
    outputs = []
    for feat, cls_convs, gfl_cls in zip(
        feats,
        self.cls_convs,
        self.gfl_cls,
    ):
        for conv in cls_convs:
            feat = conv(feat)
        output = gfl_cls(feat)
        outputs.append(output)
        '''cls_pred, reg_pred = output.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=1
        )
        cls_pred = cls_pred.sigmoid()
        out = torch.cat([cls_pred, reg_pred], dim=1)
        outputs.append(out.flatten(start_dim=2))'''
    # return torch.cat(outputs, dim=2).permute(0, 2, 1)
    return outputs
def forward(self, feats):
    if torch.onnx.is_in_onnx_export():
        return self._forward_onnx(feats)
    outputs = []
    for feat, cls_convs, gfl_cls in zip(
        feats,
        self.cls_convs,
        self.gfl_cls,
    ):
        for conv in cls_convs:
            feat = conv(feat)
        output = gfl_cls(feat)
        outputs.append(output)
        #outputs.append(output.flatten(start_dim=2))
    #outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
    return outputs
```

执行以下命令转换出onnx模型以及torchscript模型

```bash
python tools/export_torchscript.py --cfg_path config/nanodet-plus-m-1.5x_416.yml --model_path models/nanodet-plus-m-1.5x_416.pth --input_shape 416,416 --out_path nanodet_plus_m_1.5x-416.torchscript.pth

python tools/export_onnx.py --cfg_path config/nanodet-plus-m-1.5x_416.yml --model_path models/nanodet-plus-m-1.5x_416.pth --out_path nanodet_plus_m_1.5x-416.onnx
```

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [official_nanodet.yaml](../build_in/build/official_nanodet.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd nanodet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_nanodet.yaml
    ```

### step.4 模型推理

1. runstream

    - 参考[nanodet_vsx.py](../build_in/vsx/python/nanodet_vsx.py)生成预测的txt结果

    ```
    python ../build_in/vsx/python/nanodet_vsx.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/official_nanodet_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-nanodet_plus_m-vdsp_params.json \
        --label_txt path/to/coco.txt \
        --save_dir ./runstream_output \
        --device 0
    ```

    - 参考[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt runstream_output
    ```

    <details><summary>点击查看精度统计结果</summary>

    ```
    # fp16
    DONE (t=5.24s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.280
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.432
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.292
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.076
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.290
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.473
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.260
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.405
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.429
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.146
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.478
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.689
    {'bbox_mAP': 0.28, 'bbox_mAP_50': 0.432, 'bbox_mAP_75': 0.292, 'bbox_mAP_s': 0.076, 'bbox_mAP_m': 0.29, 'bbox_mAP_l': 0.473, 'bbox_mAP_copypaste': '0.280 0.432 0.292 0.076 0.290 0.473'}

    # int8
    DONE (t=5.45s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.258
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.405
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.267
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.073
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.267
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.439
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.245
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.383
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.133
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.455
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.663
    {'bbox_mAP': 0.258, 'bbox_mAP_50': 0.405, 'bbox_mAP_75': 0.267, 'bbox_mAP_s': 0.073, 'bbox_mAP_m': 0.267, 'bbox_mAP_l': 0.439, 'bbox_mAP_copypaste': '0.258 0.405 0.267 0.073 0.267 0.439'}
    ```

    </details>

### step.5 性能精度测试
1. 性能测试
    - 配置[official-nanodet_plus_m-vdsp_params.json](../build_in/vdsp_params/official-nanodet_plus_m-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_nanodet_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/official-nanodet_plus_m-vdsp_params.json -i 2 p 2 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准本，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```

    - vmpp推理，获取npz结果输出
    ```bash
    vamp -m deploy_weights/official_nanodet_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/official-nanodet_plus_m-vdsp_params.json -i 2 p 2 -b 1 --datalist npz_datalist.txt --path_output npz_output
    ```
    
    - 解析vamp输出的npz文件，参考：[npz_decode.py](../build_in/npz_decode.py)
    ```bash
    python ../build_in/npz_decode.py --txt result_npz --label_txt path/to/coco.txt --input_image_dir path/to/coco_val2017 --model_size 416 416 --vamp_datalist_path npz_datalist.txt --vamp_output_dir npz_output
    ```

    - 参考：[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt result_npz
    ```