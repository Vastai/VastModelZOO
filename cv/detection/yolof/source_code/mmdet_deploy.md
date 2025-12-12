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
    
    - 修改`mmdet/models/dense_heads/yolof_head.py`中`forward_single`函数

        ```python
        def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
            cls_score = self.cls_score(self.cls_subnet(x))
            # N, _, H, W = cls_score.shape
            # cls_score = cls_score.view(N, -1, self.num_classes, H, W)

            reg_feat = self.bbox_subnet(x)
            bbox_reg = self.bbox_pred(reg_feat)
            objectness = self.object_pred(reg_feat)
            
            return cls_score, bbox_reg, objectness

            # # implicit objectness
            # objectness = objectness.view(N, -1, 1, H, W)
            # normalized_cls_score = cls_score + objectness - torch.log(
            #     1. + torch.clamp(cls_score.exp(), max=INF) +
            #     torch.clamp(objectness.exp(), max=INF))
            # normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
            # return normalized_cls_score, bbox_reg
        ```
3. 修改模型输入shape，经测试，测试shape为1280x1280可达到论文中的精度，可通过mmdeploy中配置进行修改，也可修改mmdet中config中test_pipeline修改

    ```python
    test_pipeline = [
        dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
        dict(type='Resize', scale=(1280, 1280), keep_ratio=False),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
    ]
    ```

4. 运行以下脚本，进行模型转换

    ```bash
    python tools/deploy.py configs/mmdet/detection/detection_onnxruntime_static.py ../mmdetection_2.25.3/configs/yolof/yolof_r50_c5_8x8_1x_coco.py ../mmdetection_2.25.3/models/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth demo/resources/face.png --work-dir mmdeploy_models/yolof  --device cpu --dump-info
    ```

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [mmdet_yolof.yaml](../build_in/build/mmdet_yolof.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd yolof
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/mmdet_yolof.yaml
    ```

### step.4 模型推理

- 参考[yolof_vsx.py](../build_in/vsx/python/yolof_vsx.py)生成预测的txt结果

    ```
    python ../build_in/vsx/python/yolof_vsx.py \
        --file_path path/to/coco_val2017 \
        --model_prefix_path deploy_weights/mmdet_yolof_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/mmdet-yolof-vdsp_params.json \
        --label_txt ../../common/label/coco.txt \
        --save_dir ./infer_output \
        --device 0
    ```

- 参考[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt infer_output
    ```

    ```
    # fp16
    DONE (t=5.92s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.370
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.568
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.401
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.191
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.421
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.499
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.301
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.487
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.529
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.302
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.606
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.692
    {'bbox_mAP': 0.37, 'bbox_mAP_50': 0.568, 'bbox_mAP_75': 0.401, 'bbox_mAP_s': 0.191, 'bbox_mAP_m': 0.421, 'bbox_mAP_l': 0.499, 'bbox_mAP_copypaste': '0.370 0.568 0.401 0.191 0.421 0.499'}


    # int8
    DONE (t=5.61s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.565
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.396
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.496
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.297
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.483
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.526
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.302
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.603
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.692
    {'bbox_mAP': 0.366, 'bbox_mAP_50': 0.565, 'bbox_mAP_75': 0.396, 'bbox_mAP_s': 0.188, 'bbox_mAP_m': 0.418, 'bbox_mAP_l': 0.496, 'bbox_mAP_copypaste': '0.366 0.565 0.396 0.188 0.418 0.496'}

    ```

### step.5 性能精度测试
1. 性能测试
    ```bash
    vamp -m deploy_weights/mmdet_yolof_int8/mod --vdsp_params ../build_in/vdsp_params/mmdet-yolof-vdsp_params.json -i 1 -b 1 -d 0 -p 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```

    - vamp推理，获取npz结果输出
    ```bash
    vamp -m deploy_weights/mmdet_yolof_int8/mod --vdsp_params vdsp_params.json -i 1 -b 1 -d 0 -p 1 --datalist npz_datalist.txt --path_output npz_output
    ```

    - npz数据解析，参考：[npz_decode.py](../build_in/npz_decode.py)
    ```bash
    python ../build_in/npz_decode.py --txt result_npz --label_txt /path/to/coco.txt --input_image_dir /path/to/coco_val2017 --vamp_datalist_path npz_datalist.txt --vamp_output_dir npz_output
    ```
    
    - 参考：[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt result_npz
    ```
