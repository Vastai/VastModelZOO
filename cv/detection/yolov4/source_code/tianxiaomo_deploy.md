## tianxiaomo yolov4

### step.1 获取预训练模型
```
link: https://github.com/Tianxiaomo/pytorch-YOLOv4
branch: master
commit: a65d219f9066bae4e12003bd7cdc04531860c672
```

- 端到端，基于源仓库脚本：[demo_pytorch2onnx.py](https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/demo_pytorch2onnx.py)，执行以下命令，转换为batch维度是动态的onnx：
    ```bash
    python demo_pytorch2onnx.py weights/yolov4.pth data/dog.jpg -1 80 416 416
    ```
> Tips
> - 如需forward，将本地[tool](./tianxiaomo/tool/)文件内的文件，替换原始[tool](https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/master/tool)内对应文件。
> - 修改[demo_pytorch2onnx.py#L17](https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/demo_pytorch2onnx.py#L17)，修改参数`inference=False`，执行上面脚本，导出forward，无后处理的onnx
    
### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [tianxiaomo_config.yaml](../build_in/build/tianxiaomo_config.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd yolov4
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/tianxiaomo_config.yaml
    ```

### step.4 模型推理
推理：[tianxiaomo_yolov4_detector.py](../build_in/vsx/tianxiaomo_yolov4_detector.py)
    - 配置模型路径和测试数据路径等参数

    ```
    python ../build_in/vsx/tianxiaomo_yolov4_detector.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/tianxiaomo_yolov4_e2e_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/tianxiaomo-yolov4-vdsp_params.json \
        --label_txt ../../common/label/coco.txt \
        --save_dir ./infer_output \
        --device 0
    ```

    - 精度评估，参考：[eval_map.py](../../common/eval/eval_map.py)
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt ./infer_output
    ```

    <details><summary>点击查看精度测试结果</summary>
    
    ```
    # 模型名：yolov4-416

    # fp16
    DONE (t=2.73s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.433
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.657
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.470
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.212
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.494
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.623
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.528
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.564
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.326
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.638
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.765
    {'bbox_mAP': 0.433, 'bbox_mAP_50': 0.657, 'bbox_mAP_75': 0.47, 'bbox_mAP_s': 0.212, 'bbox_mAP_m': 0.494, 'bbox_mAP_l': 0.623, 'bbox_mAP_copypaste': '0.433 0.657 0.470 0.212 0.494 0.623'}

    # int8
    DONE (t=2.78s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.424
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.646
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.461
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.202
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.616
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.323
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.520
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.557
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.313
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.636
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.755
    {'bbox_mAP': 0.424, 'bbox_mAP_50': 0.646, 'bbox_mAP_75': 0.461, 'bbox_mAP_s': 0.202, 'bbox_mAP_m': 0.484, 'bbox_mAP_l': 0.616, 'bbox_mAP_copypaste': '0.424 0.646 0.461 0.202 0.484 0.616'}

    ```

    </details>

### step.5 性能精度测试
1. 性能测试
    - 配置[tianxiaomo-yolov4-vdsp_params.json](../build_in/vdsp_params/tianxiaomo-yolov4-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/tianxiaomo_yolov4_e2e_fp16/mod --vdsp_params ../build_in/vdsp_params/tianxiaomo-yolov4-vdsp_params.json -i 1 p 1 -b 1 -d 0
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path path/to/coco_val2017 \
        --target_path  path/to/coco_val2017_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理获取npz结果输出
    ```bash
    vamp -m deploy_weights/tianxiaomo_yolov4_e2e_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/tianxiaomo-yolov4-vdsp_params.json \
        -i 1 p 1 -b 1 \
        --datalist path/to/npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz文件，参考：[npz_decode.py](../../common/utils/npz_decode.py)
    ```bash
    python ../../common/utils/npz_decode.py \
        --txt result_npz --label_txt ../../common/label/coco.txt \
        --input_image_dir path/to/coco_val2017 \
        --model_size 640 640 \
        --vamp_datalist_path path/to/npz_datalist.txt \
        --vamp_output_dir npz_output
    ```

    - 精度统计，参考：[eval_map.py](../../common/eval/eval_map.py)
    ```bash
    python ../../common/eval/eval_map.py \
        --gt path/to/instances_val2017.json \
        --txt path/to/result_npz
    ```

