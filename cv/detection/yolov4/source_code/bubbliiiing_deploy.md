## bubbliiiing yolov4

### step.1 获取预训练模型
```

link: https://github.com/bubbliiiing/yolov4-tiny-pytorch
branch: master
commit: 60598a259cfe64b4b87f064fbd4b183a4cdd6cba

link: https://github.com/bubbliiiing/yolov4-pytorch
branch: master
commit: b7c2212250037c262282bac06fcdfe97ac86c055
```
- 克隆原始yolov4仓库或yolov4仓库，将所有文件移动至[yolov4/source_code/bubbliiiing](../source_code/bubbliiiing)文件夹下
- 修改[yolo.py#L20](https://github.com/bubbliiiing/yolov4-pytorch/blob/master/yolo.py#L20)，yolo类中的`_defaults`参数（配置文件路径，开启letterbox，关闭cuda）
- 基于[onnx_convert.py](../source_code/bubbliiiing/onnx_convert.py)，导出onnx


### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)

### step.3 模型转换
1. 参考瀚博训推软件生态链文档，获取模型转换工具: [vamc v3.0+](../../../../docs/vastai_software.md)

2. 根据具体模型，修改编译配置
    - [bubbliiiing_config.yaml](../build_in/build/bubbliiiing_config.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

3. 模型编译

    ```bash
    cd yolov4
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/bubbliiiing_config.yaml
    ```
> Tips:
> 
> bubbliiiing来源的yolov4和yolov4_tiny均只支持`forward`推理，配置表的`extra_ops`参数设置为`type: null`
>


### step.4 模型推理
1. 参考瀚博训推软件生态链文档，获取模型推理工具：[vaststreamx v2.8+](../../../../docs/vastai_software.md)

2. runstream推理：[bubbliiiing_yolov4_detector.py](../build_in/vsx/bubbliiiing_yolov4_detector.py)
    - 配置模型路径和测试数据路径等参数

    ```
    python ../build_in/vsx/bubbliiiing_yolov4_detector.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/bubbliiiing_yolov4_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/bubbliiiing-yolov4-vdsp_params.json \
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
    DONE (t=2.34s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.428
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.648
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.464
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.196
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.489
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.619
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.510
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.545
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.292
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.618
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.745
    {'bbox_mAP': 0.428, 'bbox_mAP_50': 0.648, 'bbox_mAP_75': 0.464, 'bbox_mAP_s': 0.196, 'bbox_mAP_m': 0.489, 'bbox_mAP_l': 0.619, 'bbox_mAP_copypaste': '0.428 0.648 0.464 0.196 0.489 0.619'}

    # int8
    DONE (t=2.34s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.419
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.636
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.455
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.478
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.611
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.502
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.537
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.280
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.615
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.736
    {'bbox_mAP': 0.419, 'bbox_mAP_50': 0.636, 'bbox_mAP_75': 0.455, 'bbox_mAP_s': 0.188, 'bbox_mAP_m': 0.478, 'bbox_mAP_l': 0.611, 'bbox_mAP_copypaste': '0.419 0.636 0.455 0.188 0.478 0.611'}
    ```

    </details>


### step.5 性能精度
1. 参考瀚博训推软件生态链文档，获取模型性能测试工具：[vamp v2.4+](../../../../docs/vastai_software.md)

2. 性能测试
    - 配置[bubbliiiing-yolov4-vdsp_params.json](../build_in/vdsp_params/bubbliiiing-yolov4-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/bubbliiiing_yolov4_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/bubbliiiing-yolov4-vdsp_params.json -i 1 p 1 -b 1 -d 0
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
    vamp -m deploy_weights/bubbliiiing_yolov4_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/bubbliiiing-yolov4-vdsp_params.json \
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
