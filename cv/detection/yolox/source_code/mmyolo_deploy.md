### step.1 获取预训练模型
```bash
cd mmyolo
python ./projects/easydeploy/tools/export.py \
	configs/yolox/yolox_s-v61_syncbn_fast_8xb16-300e_coco.py \
	yoloxs.pth \
	--work-dir work_dir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 1 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)

### step.3 模型转换
1. 根据具体模型修改模型转换配置文件
    - [mmyolo_yolox.yaml](../build_in/build/mmyolo_yolox.yaml)

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`

2. 编译模型
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd yolovx
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/mmyolo_yolox.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights/mmyolo_yolovx_run_stream_int8`文件夹，其中包含转换后的模型文件。

### step.4 模型推理
1. 可以利用[脚本](../build_in/vsx/python/mmyolo_yolox_runstream.py)生成预测的txt结果

    ```
    python ../build_in/vsx/python/mmyolo_yolox_runstream.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/mmyolo_yolox_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/mmyolo-yolox_s-vdsp_params.json \
        --label_txt path/to/coco.txt \
        --save_dir ./runstream_output \
        --device 0
    ```
    - 注意替换命令行中--file_path和--label_txt为实际路径

2. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
    ```bash
        python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt ./runstream_output
    ```
    - 测试精度数据如下：
    ```
    DONE (t=2.71s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.378
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.581
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.404
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.200
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.419
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.506
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.314
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.492
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.315
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.564
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.667
    {'bbox_mAP': 0.378, 'bbox_mAP_50': 0.581, 'bbox_mAP_75': 0.404, 'bbox_mAP_s': 0.2, 'bbox_mAP_m': 0.419, 'bbox_mAP_l': 0.506, 'bbox_mAP_copypaste': '0.378 0.581 0.404 0.200 0.419 0.506'}

    ```

### step.6 性能精度测试
1. 性能测试
    ```bash
    vamp -m deploy_weights/mmyolo_yolox_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/mmyolo-yolox_s-vdsp_params.json -i 2 p 2 -b 1
    ```


2. 精度测试
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
    vamp -m deploy_weights/mmyolo_yolox_run_stream_int8/mod \
        --vdsp_params ./build_in/vdsp_params/mmyolo-yolox_s-vdsp_params.json \
        -i 2 p 2 -b 1 \
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
    
    - 参考：[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py \
        --gt path/to/instances_val2017.json \
        --txt path/to/vamp_draw_output
    ```

