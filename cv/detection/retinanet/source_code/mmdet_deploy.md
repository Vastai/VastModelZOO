# mmdet

### step.1 获取预训练模型

1. 安装mmdetection、mmdeploy、mmcv等依赖，mmdet3.x新版需搭配mmdeploy进行模型转换
2. 目前vacc部署不支持yolof后处理，因此需对源代码修改，去除后处理生成onnx模型
    - 修改`mmdet/models/dense_heads/base_dense_head.py`中`predict`函数
        ```python
        def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]

            outs = self(x)
            return outs
        ```
    
3. 修改模型输入shape，经测试，测试shape为1280x1280可达到论文中的精度，可通过mmdeploy中配置进行修改，也可修改mmdet中config中test_pipeline修改

    ```python
    test_pipeline = [
        dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
        dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
    ]
    ```

4. 运行以下脚本，进行模型转换

    ```bash
    python tools/deploy.py configs/mmdet/detection/detection_onnxruntime_static.py ../mmdetection/configs/retinanet/retinanet_r50-caffe_fpn_1x_coco.py   ../mmdetection/models/retinanet/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth demo/resources/face.png --work-dir mmdeploy_models/retinanet/retinanet_r50_caffe_fpn_1x_coco  --device cpu --dump-info
    ```

### step.2 准备数据集

- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [mmdet_retinanet.yaml](../build_in/build/mmdet_retinanet.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd retinanet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/mmdet_retinanet.yaml
    ```

### step.4 模型推理

- 参考[mmdet_retinanet_vsx.py](../build_in/vsx/python/mmdet_retinanet_vsx.py)生成预测的txt结果

    ```
    python ../build_in/vsx/python/mmdet_retinanet_vsx.py \
        --file_path path/to/coco_val2017 \
        --model_prefix_path deploy_weights/mmdet_retinanet_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/mmdet-retinanet_r50_fpn_1x_coco-vdsp_params.json \
        --label_txt path/to/coco.txt \
        --save_dir ./infer_output \
        --device 0
    ```

- 参考[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt infer_output
    ```

    <details><summary>点击查看精度测试结果</summary>

    ```
    # fp16
    DONE (t=9.97s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.349
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.520
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.383
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.183
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.396
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.468
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.301
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.491
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.553
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.326
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.605
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.701
    {'bbox_mAP': 0.349, 'bbox_mAP_50': 0.52, 'bbox_mAP_75': 0.383, 'bbox_mAP_s': 0.183, 'bbox_mAP_m': 0.396, 'bbox_mAP_l': 0.468, 'bbox_mAP_copypaste': '0.349 0.520 0.383 0.183 0.396 0.468'}

    # int8
    DONE (t=9.90s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.346
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.516
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.379
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.395
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.465
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.490
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.551
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.321
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.607
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.697
    {'bbox_mAP': 0.346, 'bbox_mAP_50': 0.516, 'bbox_mAP_75': 0.379, 'bbox_mAP_s': 0.179, 'bbox_mAP_m': 0.395, 'bbox_mAP_l': 0.465, 'bbox_mAP_copypaste': '0.346 0.516 0.379 0.179 0.395 0.465'}
    ```

    </details>

### step.5 性能精度测试

1. 性能测试
    ```bash
    vamp -m deploy_weights/mmdet_retinanet_int8/mod --vdsp_params ../build_in/vdsp_params/mmdet-retinanet_r50_fpn_1x_coco-vdsp_params.json -i 1 -b 1 -d 0 -p 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；
    
    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path path/to/coco_val2017 \
        --target_path  path/to/coco_val2017_npz  \
        --text_path npz_datalist.txt
    ```
    
    - vamp推理，获取npz结果输出
    ```bash
    vamp -m deploy_weights/mmdet_retinanet_int8/mod \
        --vdsp_params ../build_in/vdsp_params/mmdet-retinanet_r50_fpn_1x_coco-vdsp_params.json \
        -i 1 -b 1 -d 0 -p 1 \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 精度统计，参考：[eval_map.py](../../common/eval/eval_map.py)，
    ```bash
    python ../../common/eval/eval_map.py \
        --gt path/to/instances_val2017.json 
        --txt npz_output
    ```