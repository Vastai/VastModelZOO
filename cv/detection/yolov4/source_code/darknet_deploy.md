## darknet yolov4


### step.1 获取预训练模型

vamc支持直接通过darknet模型转换为三件套，只需从[项目](https://github.com/AlexeyAB/darknet)中拉取原始darknet模型即可

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换

1. 参考瀚博训推软件生态链文档，获取模型转换工具: [vamc v3.0+](../../../../docs/vastai_software.md)

2. 根据具体模型，修改编译配置
    - [darknet_config.yaml](../build_in/build/darknet_config.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

3. 模型编译

    ```bash
    cd yolov4
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/darknet_config.yaml
    ```
> Tips:
> 
> 注意需要将下载好的yolov4.cfg文件提前放入${workspace.path}/${name}路径下
>

### step.4 模型推理
1. 参考瀚博训推软件生态链文档，获取模型推理工具：[vaststreamx v2.8+](../../../../docs/vastai_software.md)

2. runstream推理：[darknet_yolov4_detector.py](../build_in/vsx/darknet_yolov4_detector.py)
    - 配置模型路径和测试数据路径等参数

    ```
    python ../build_in/vsx/darknet_yolov4_detector.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/darknet_yolov4_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/darknet-yolov4-vdsp_params.json \
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
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.472
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.702
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.517
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.272
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.531
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.636
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.350
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.564
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.603
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.387
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.670
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.780
    {'bbox_mAP': 0.472, 'bbox_mAP_50': 0.702, 'bbox_mAP_75': 0.517, 'bbox_mAP_s': 0.272, 'bbox_mAP_m': 0.531, 'bbox_mAP_l': 0.636, 'bbox_mAP_copypaste': '0.472 0.702 0.517 0.272 0.531 0.636'}

    # int8
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.440
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.686
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.486
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.242
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.488
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.606
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.332
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.538
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.578
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.364
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.642
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.751
    {'bbox_mAP': 0.44, 'bbox_mAP_50': 0.686, 'bbox_mAP_75': 0.486, 'bbox_mAP_s': 0.242, 'bbox_mAP_m': 0.488, 'bbox_mAP_l': 0.606, 'bbox_mAP_copypaste': '0.440 0.686 0.486 0.242 0.488 0.606'}

    ```

    </details>

### step.5 性能测试
1. 参考瀚博训推软件生态链文档，获取模型性能测试工具：[vamp v2.4+](../../../../docs/vastai_software.md)

2. 性能测试
    - 配置[darknet-yolov4-vdsp_params.json](../build_in/vdsp_params/darknet-yolov4-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/darknet_yolov4_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/darknet-yolov4-vdsp_params.json -i 1 p 1 -b 1 -d 0
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
    vamp -m deploy_weights/darknet_yolov4_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/darknet-yolov4-vdsp_params.json \
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
