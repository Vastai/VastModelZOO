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
> - 如需forward，将本地[tool](../source_code/tianxiaomo/tool)文件内的文件，替换原始[tool](https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/master/tool)内对应文件。
> - 修改[demo_pytorch2onnx.py#L17](https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/demo_pytorch2onnx.py#L17)，修改参数`inference=False`，执行上面脚本，导出forward，无后处理的onnx
    
### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换

1. 参考瀚博训推软件生态链文档，获取模型转换工具: [vamc v3.0+](../../../../docs/vastai_software.md)

2. 根据具体模型，修改编译配置
    - [tianxiaomo_config.yaml](../build_in/build/tianxiaomo_config.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

3. 模型编译

    ```bash
    cd yolov4
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/tianxiaomo_config.yaml
    ```

### step.4 模型推理

1. 参考瀚博训推软件生态链文档，获取模型推理工具：[vaststreamx v2.8+](../../../../docs/vastai_software.md)

2. runstream推理：[tianxiaomo_yolov4_detector.py](../build_in/vsx/tianxiaomo_yolov4_detector.py)
    - 配置模型路径和测试数据路径等参数

    ```
    python ../build_in/vsx/tianxiaomo_yolov4_detector.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/tianxiaomo_yolov4_e2e_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/tianxiaomo-yolov4-vdsp_params.json \
        --label_txt path/to/coco.txt \
        --save_dir ./runstream_output \
        --device 0
    ```

    - 精度评估，参考：[eval_map.py](../../common/eval/eval_map.py)
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt ./runstream_output
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

### step.5 性能测试

1. 参考瀚博训推软件生态链文档，获取模型性能测试工具：[vamp v2.4+](../../../../docs/vastai_software.md)

2. 性能测试
    - 配置[tianxiaomo-yolov4-vdsp_params.json](../build_in/vdsp_params/tianxiaomo-yolov4-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/tianxiaomo_yolov4_e2e_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/tianxiaomo-yolov4-vdsp_params.json -i 1 p 1 -b 1 -d 0
    ```

    <details><summary>点击查看性能测试结果</summary>

    ```
    # fp16
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    samples: 1024
    forwad time (s): 8.24034
    throughput (qps): 124.267
    ai utilize (%): 96.3077
    die memory used (MB): 883.234
    e2e latency (us):
        avg latency: 135770
        min latency: 16622
        max latency: 148289
    model latency (us):
        avg latency: 7749
        min latency: 7749
        max latency: 7749

    # int8
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    samples: 1024
    forwad time (s): 2.79899
    throughput (qps): 365.846
    ai utilize (%): 96.0769
    die memory used (MB): 819.371
    e2e latency (us):
        avg latency: 46046
        min latency: 5943
        max latency: 60312
    model latency (us):
        avg latency: 2625
        min latency: 2625
        max latency: 2625

    # 硬件信息
    Smi version:3.2.1
    SPI production for Bbox mode information of
    =====================================================================
    Appointed Entry:0 Device_Id:0 Die_Id:0 Die_Index:0x00000000
    ---------------------------------------------------------------------
    #               Field Name                    Value
    0              FileVersion                       V2
    1                 CardType                  VA1-16G
    2                      S/N             FCA129E00172
    3                 BboxMode              Highperf-AI
    =====================================================================
    =====================================================================
    Appointed Entry:0 Device_Id:0 Die_Id:0 Die_Index:0x00000000
    ---------------------------------------------------------------------
    OCLK:       880 MHz    ODSPCLK:    835 MHz    VCLK:       300 MHz    
    ECLK:        20 MHz    DCLK:        20 MHz    VDSPCLK:    900 MHz    
    UCLK:      1067 MHz    V3DCLK:     100 MHz    CCLK:      1000 MHz    
    XSPICLK:     50 MHz    PERCLK:     200 MHz    CEDARCLK:   500 MHz
    ```

    </details>

3. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path path/to/coco_val2017 \
        --target_path  path/to/coco_val2017_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理获取npz结果输出
    ```bash
    vamp -m deploy_weights/tianxiaomo_yolov4_e2e_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/tianxiaomo-yolov4-vdsp_params.json \
        -i 1 p 1 -b 1 \
        --datalist datasets/coco_npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz文件，参考：[npz_decode.py](../../common/utils/npz_decode.py)
    ```bash
    python ../../common/utils/npz_decode.py \
        --txt result_npz --label_txt datasets/coco.txt \
        --input_image_dir datasets/coco_val2017 \
        --model_size 640 640 \
        --vamp_datalist_path datasets/coco_npz_datalist.txt \
        --vamp_output_dir npz_output
    ```

    - 精度统计，参考：[eval_map.py](../../common/eval/eval_map.py)
    ```bash
    python ../../common/eval/eval_map.py \
        --gt path/to/instances_val2017.json \
        --txt path/to/vamp_draw_output
    ```

