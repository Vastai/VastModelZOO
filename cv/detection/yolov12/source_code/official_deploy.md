### step.1 获取预训练模型

```
link: https://github.com/sunsmarterjie/yolov12.git
commit: 3a336a4adf3683d280e1a50d03fa24bbe7f24a5b
```

- 获取原始仓库，按原仓库安装
- 参考[export_onnx.py](./export_onnx.py)，导出onnx

### step.2 准备数据集
- [校准数据集](http://192.168.20.139:8888/vastml/dataset/det/COCO2017/det_coco_calib/?download=zip)
- [评估数据集](http://192.168.20.139:8888/vastml/dataset/det/COCO2017/det_coco_val/?download=zip)
- [gt: instances_val2017.json](http://192.168.20.139:8888/vastml/dataset/det/COCO2017/instances_val2017.json)
- [label: coco.txt](http://192.168.20.139:8888/vastml/dataset/det/COCO2017/coco.txt)


### step.3 模型转换

1. 获取[vamc](../../../docs/doc_vamc.md)模型转换工具

2. 根据具体模型，修改编译配置
    - [official_yolov12.yaml](../vacc_code/build/official_yolov12.yaml)
    
    > - runmodel推理，编译参数`backend.type: tvm_runmodel`
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

3. 模型编译
    ```bash
    cd yolov12
    mkdir workspace
    cd workspace
    vamc compile ../vacc_code/build/official_yolov12.yaml
    ```

### step.4 模型推理

1. runmodel
    - 参考：[runmodel.py](../vacc_code/runmodel/runmodel.py)
    ```
    python ../vacc_code/runmodel/runmodel.py \
        --file_path /path/to/det_coco_val \
        --model_weight_path deploy_weights/official_yolov12_run_model_fp16/  \
        --model_name mod \
        --model_input_name images \
        --model_input_shape 1,3,640,640 \
        --label_txt /path/to/coco.txt \
        --save_dir ./runmodel_output 
    ```

2. runstream
    - 参考[yolov12_vsx.py](../vacc_code/vsx/python/yolov12_vsx.py)生成预测的txt结果

    ```
    python ../vacc_code/vsx/python/yolov12_vsx.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/official_yolov12_run_stream_fp16/mod \
        --vdsp_params_info ../vacc_code/vdsp_params/official-yolov12s-vdsp_params.json \
        --label_txt path/to/coco.txt \
        --save_dir ./runstream_output \
        --device 0
    ```

    - 参考[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt path/to/result
    ```

    <details><summary>点击查看精度测试结果</summary>

    ```
    # 模型名：yolov12s-640

    # fp16
    DONE (t=2.51s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.462
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.628
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.497
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.263
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.512
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.645
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.354
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.562
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.374
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.650
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.777
    {'bbox_mAP': 0.462, 'bbox_mAP_50': 0.628, 'bbox_mAP_75': 0.497, 'bbox_mAP_s': 0.263, 'bbox_mAP_m': 0.512, 'bbox_mAP_l': 0.645, 'bbox_mAP_copypaste': '0.462 0.628 0.497 0.263 0.512 0.645'}

    ```

    </details>


### step.5 性能精度
1. 获取[vamp](../../../docs/doc_vamp.md)工具

2. 性能测试
    ```bash
    vamp -m deploy_weights/official_yolov12_run_stream_fp16/mod --vdsp_params ../vacc_code/vdsp_params/official-yolov12s-vdsp_params.json -i 1 -b 1 -d 0 -p 1
    ```

    <details><summary>点击查看性能测试结果</summary>

    ```bash
    # fp16
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    samples: 1024
    forwad time (s): 15.5398
    throughput (qps): 65.8953
    ai utilize (%): 95.8243
    die memory used (MB): 835.613
    e2e latency (us):
        avg latency: 255913
        min latency: 16150
        max latency: 259072
    model latency (us):
        avg latency: 14541
        min latency: 14541
        max latency: 14541

    
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
        --target_path  path/to/coco_val2017_npz  \
        --text_path npz_datalist.txt
    ```

    - vamp推理，获取npz结果输出
    ```bash
    vamp -m deploy_weights/official_yolov12_run_stream_fp16/mod \
        --vdsp_params ../vacc_code/vdsp_params/official-yolov12s-vdsp_params.json \
        -i 1 -b 1 -d 0 -p 1 \
        --datalist datasets/coco_npz_datalist.txt \
        --path_output npz_output
    ```

    - npz数据解析，参考：[vamp_npz_decode.py](../../common/eval/vamp_npz_decode.py)
    ```bash
    python vamp_npz_decode.py \
        --label_txt ../../eval/coco.txt \
        --input_image_dir ~/datasets/detection/coco_val2017 \
        --vamp_datalist_path ~/datasets/detection/datalist_npz.txt \
        --vamp_output_dir npz_output
    ```
    
    - 参考：[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py \
        --gt path/to/instances_val2017.json \
        --txt path/to/vamp_draw_output
    ```

### Tips
- 当前仅提供fp16的模型，后续会提供int8模型
- yolov8和yolov12可以使用同一个的后处理算子
- yolov12的支持需要更新AI包，包名为：ai-v2.8.1-20250428-742-linux-x86_64-sdk2-python3.8.bin，后续会有正式包发布