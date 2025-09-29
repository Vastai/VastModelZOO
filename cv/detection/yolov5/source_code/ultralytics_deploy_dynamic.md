## ultralytics yolov5 dynamic

### step.1 获取预训练模型

1. v6.1

    基于分支v6.0/v6.1通过原始项目提供的脚本[export.py](https://github.com/ultralytics/yolov5/blob/v6.1/export.py)转换至torchscript或onnx模型

2. v7.0

    基于分支v7.0可以修改项目中[此行代码](https://github.com/ultralytics/yolov5/blob/v7.0/models/yolo.py#L79)如下
    ```python
    return torch.cat(z, 1), x
    ```
    然后运行如下命令转换生成onnx模型
    ```bash
    python export.py --weights yolov5s.pt --include onnx 
    ```

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [ultralytics_dynamic_yolov5.yaml](../build_in/build/ultralytics_dynamic_yolov5.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd yolov5
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ultralytics_dynamic_yolov5.yaml
    ```

### step.4 模型推理
1. runstream推理：[yolov5_ultralytics_dynamic.py](../build_in/vsx/yolov5_ultralytics_dynamic.py)
    - 配置模型路径和测试数据路径等参数

    ```
    python ../build_in/vsx/yolov5_ultralytics_dynamic.py \
        --dataset_root path/to/coco_val2017 \
        --dataset_filelist path/to/coco_val2017 \
        --module_info deploy_weights/ultralytics_yolov5s_dynamic_run_stream_int8/ultralytics_yolov5s_dynamic_run_stream_int8_module_info.json \
        --vdsp_params ../build_in/vdsp_params/yolo_div255_bgr888.json \
        --label_file ../../common/label/coco.txt \
        --dataset_output_folder ./runstream_output \
        --device 0
    ```

    - 精度评估，参考：[eval_map.py](../../common/eval/eval_map.py)
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt ./runstream_output
    ```

    <details><summary>点击查看精度测试结果</summary>
    
    ```
    # 模型名：yolov5s-640-dynamic

    # fp16
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.370
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.561
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.400
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.207
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.490
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.304
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.494
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.530
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.327
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.592
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.682
    {'bbox_mAP': 0.37, 'bbox_mAP_50': 0.561, 'bbox_mAP_75': 0.4, 'bbox_mAP_s': 0.207, 'bbox_mAP_m': 0.42, 'bbox_mAP_l': 0.49, 'bbox_mAP_copypaste': '0.370 0.561 0.400 0.207 0.420 0.490'}

    # int8
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.353
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.554
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.384
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.197
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.394
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.476
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.293
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.477
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.313
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.569
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.667
    {'bbox_mAP': 0.353, 'bbox_mAP_50': 0.554, 'bbox_mAP_75': 0.384, 'bbox_mAP_s': 0.197, 'bbox_mAP_m': 0.394, 'bbox_mAP_l': 0.476, 'bbox_mAP_copypaste': '0.353 0.554 0.384 0.197 0.394 0.476'}
    ```

    </details>


### step.5 性能精度测试
1. 动态尺寸模型不能使用vamp工具测试性能，需要使用性能测试脚本进行测试，参考：[dynamic_yolo_prof.py](../build_in/vsx/dynamic_yolo_prof.py)
    ```bash
    python3 ../build_in/vsx/dynamic_yolo_prof.py \ 
        -m ./deploy_weights/ultralytics_dynamic_yolov5_run_stream_int8/ultralytics_dynamic_yolov5_run_stream_int8_module_info.json \
        --vdsp_params ../build_in/vdsp_params/yolo_div255_bgr888.json \
        --max_input_shape "[1,3,640,640]" \
        --model_input_shape "[1,3,640,640]" \
        --device_ids [0] \
        --batch_size 2 \
        --instance 1 \
        --iterations 5000 \
        --shape "[3,640,640]" \
        --percentiles "[50,90,95,99]" \
        --input_host 1 \
        --queue_size 1
    ```
