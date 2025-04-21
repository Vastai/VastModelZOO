## ultralytics yolov5

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
1. 参考瀚博训推软件生态链文档，获取模型转换工具: [vamc v3.0+](../../../../docs/vastai_software.md)

2. 根据具体模型，修改编译配置
    - [ultralytics_yolov5.yaml](../build_in/build/ultralytics_yolov5.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

3. 模型编译

    ```bash
    cd yolov5
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ultralytics_yolov5.yaml
    ```

### step.4 模型推理

1. 参考瀚博训推软件生态链文档，获取模型推理工具：[vaststreamx v2.8+](../../../../docs/vastai_software.md)

2. runstream推理：[detection.py](../../common/vsx/detection.py)
    - 配置模型路径和测试数据路径等参数

    ```
    python ../../common/vsx/detection.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/ultralytics_yolov5s_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/ultralytics-yolov5s-vdsp_params.json \
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
    # 模型名：yolov5s-640

    # fp16
    DONE (t=3.79s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.370
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.561
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.399
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.209
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.421
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.487
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.304
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.496
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.532
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.331
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.591
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.682
    {'bbox_mAP': 0.37, 'bbox_mAP_50': 0.561, 'bbox_mAP_75': 0.399, 'bbox_mAP_s': 0.209, 'bbox_mAP_m': 0.421, 'bbox_mAP_l': 0.487, 'bbox_mAP_copypaste': '0.370 0.561 0.399 0.209 0.421 0.487'}

    # int8
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.353
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.550
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.383
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.197
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.398
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.460
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.294
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.482
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.520
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.327
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.576
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.666
    {'bbox_mAP': 0.353, 'bbox_mAP_50': 0.55, 'bbox_mAP_75': 0.383, 'bbox_mAP_s': 0.197, 'bbox_mAP_m': 0.398, 'bbox_mAP_l': 0.46, 'bbox_mAP_copypaste': '0.353 0.550 0.383 0.197 0.398 0.460'}
    ```

    </details>

### step.5 性能精度
1. 参考瀚博训推软件生态链文档，获取模型性能测试工具：[vamp v2.4+](../../../../docs/vastai_software.md)

2. 性能测试
    - 配置[ultralytics-yolov5s-vdsp_params.json](../build_in/vdsp_params/ultralytics-yolov5s-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/ultralytics_yolov5s_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/ultralytics-yolov5s-vdsp_params.json -i 1 p 1 -b 1 -d 0
    ```

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
    vamp -m deploy_weights/ultralytics_yolov5s_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/ultralytics-yolov5s-vdsp_params.json \
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
