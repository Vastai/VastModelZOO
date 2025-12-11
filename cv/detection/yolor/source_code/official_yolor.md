### step.1 获取预训练模型

yolor官方项目提供了模型转换脚本，可以执行以下命令进行转换。目前onnx格式转换仍有部分问题，因此仅需转换torchscript格式即可

```bash
python models/export.py --weights weights/yolor-w6.pt --img-size 640 640
```

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [official_yolor.yaml](../build_in/build/official_yolor.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd yolor
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_yolor.yaml
    ```

### step.4 模型推理


    - 参考：[detection.py](../../common/vsx/python/detection.py)
    ```bash
    python ../../common/vsx/python/detection.py \
        --file_path path/to/coco_val2017 \
        --model_prefix_path deploy_weights/official_yolor_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-yolor_d6-vdsp_params.json \
        --label_txt path/to/coco.txt \
        --save_dir ./infer_output \
        --device 0
    ```

    - 参考[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt infer_output
    ```

    <details><summary>点击查看精度统计结果</summary>

    ```
    # fp16
    DONE (t=1.60s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.460
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.623
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.499
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.243
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.515
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.660
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.355
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.521
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.529
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.281
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.589
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.743
    {'bbox_mAP': 0.46, 'bbox_mAP_50': 0.623, 'bbox_mAP_75': 0.499, 'bbox_mAP_s': 0.243, 'bbox_mAP_m': 0.515, 'bbox_mAP_l': 0.66, 'bbox_mAP_copypaste': '0.460 0.623 0.499 0.243 0.515 0.660'}

    # int8
    DONE (t=1.59s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.455
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.620
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.496
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.244
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.507
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.654
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.351
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.515
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.523
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.284
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.578
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.738
    {'bbox_mAP': 0.455, 'bbox_mAP_50': 0.62, 'bbox_mAP_75': 0.496, 'bbox_mAP_s': 0.244, 'bbox_mAP_m': 0.507, 'bbox_mAP_l': 0.654, 'bbox_mAP_copypaste': '0.455 0.620 0.496 0.244 0.507 0.654'}
    ```

    </details>


### step.5 性能精度测试
1. 性能测试
    配置vdsp参数[official-yolor_w6-vdsp_params.json](../build_in/vdsp_params/official-yolor_w6-vdsp_params.json)：
    ```
    vamp -m deploy_weights/official_yolor_fp16/mod --vdsp_params ../build_in/vdsp_params/official-yolor_w6-vdsp_params.json -i 2 p 2 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
    
    - vamp推理，获得npz结果
    ```bash
    vamp -m deploy_weights/official_yolor_fp16/mod --vdsp_params ../build_in/vdsp_params/official-yolor_w6-vdsp_params.json -i 2 p 2 -b 1 --datalist npz_datalist.txt --path_output npz_output
    ```
    - 解析npz文件，参考：[npz_decode.py](../build_in/vdsp_params/npz_decode.py)
    ```bash
    python ../build_in/vdsp_params/npz_decode.py --txt result_npz --label_txt path/to/coco.txt --input_image_dir path/to/coco_val2017 --model_size 640 640 --vamp_datalist_path npz_datalist.txt --vamp_output_dir npz_output
    ```

    - 精度验证，参考：[eval_map.py](../../common/eval/eval_map.py)
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt result_npz
    ```

## Tips
- YOLO系列模型中，官方在精度测试和性能测试时，设定了不同的conf和iou参数
- VACC在不同测试任务中，需要分别配置build yaml内的对应参数，分别进行build模型
- `precision mode：--confidence_threshold 0.001 --nms_threshold 0.65`
- `performance mode：--confidence_threshold 0.25 --nms_threshold 0.45`