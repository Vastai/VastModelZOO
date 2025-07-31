### step.1 获取预训练模型

通过原始项目提供的脚本[export.py](https://github.com/ultralytics/yolov3/blob/v9.6.0/export.py)转换至torchscript或onnx模型

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [ultralytics_yolov3.yaml](../build_in/build/ultralytics_yolov3.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd yolov3
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ultralytics_yolov3.yaml
    ```

### step.4 模型推理
1. runstream推理：[yolov3_vsx.py](../build_in/vsx/yolov3_vsx.py)
    - 配置模型路径和测试数据路径等参数

    ```
    python ../build_in/vsx/yolov3_vsx.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/ultralytics_yolov3_run_stream_int8/mod \
        --vdsp_params_info ../vacc_code/vdsp_params/ultralytics-yolov3_tiny-vdsp_params.json \
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
    # 模型名：yolov3-416

    # fp16
    DONE (t=1.16s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.398
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.556
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.432
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.177
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.452
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.595
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.317
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.451
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.456
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.198
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.514
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.674
    {'bbox_mAP': 0.398, 'bbox_mAP_50': 0.556, 'bbox_mAP_75': 0.432, 'bbox_mAP_s': 0.177, 'bbox_mAP_m': 0.452, 'bbox_mAP_l': 0.595, 'bbox_mAP_copypaste': '0.398 0.556 0.432 0.177 0.452 0.595'}

    # int8
    DONE (t=1.51s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.548
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.420
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.167
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.441
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.585
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.308
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.438
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.443
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.186
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.499
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.664
    {'bbox_mAP': 0.387, 'bbox_mAP_50': 0.548, 'bbox_mAP_75': 0.42, 'bbox_mAP_s': 0.167, 'bbox_mAP_m': 0.441, 'bbox_mAP_l': 0.585, 'bbox_mAP_copypaste': '0.387 0.548 0.420 0.167 0.441 0.585'}
    ```

    </details>


### step.5 性能精度测试
1. 性能测试
    ```bash
    vamp -m deploy_weights/ultralytics_yolov3_run_stream_int8/mod  --vdsp_params ../vacc_code/vdsp_params/ultralytics-yolov3-vdsp_params.json -i 2 p 2 -b 1
    ```

2. 精度测试
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
    vamp -m deploy_weights/ultralytics_yolov3_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/ultralytics-yolov3-vdsp_params.json \
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
