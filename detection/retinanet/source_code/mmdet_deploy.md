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
- 准备[COCO](https://cocodataset.org/#download)数据集

### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[mmdet_config.yaml](../vacc_code/build/mmdet_config.yaml)：
    ```bash
    vamc build ../vacc_code/build/mmdet_config.yaml
    ```

### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    ```bash
    vamp -m deploy_weights/retinanet-int8-max-1_3_1024_1024-vacc/retinanet --vdsp_params ../vacc_code/vdsp_params/mmdet-retinanet_r50_fpn_1x_coco-vdsp_params.json -i 1 -b 1 -d 0 -p 1
    ```

    ```bash
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 41.463
    temperature (°C): 38.2367
    card power (W): 30.7892
    die memory used (MB): 1660.78
    throughput (qps): 42.8705
    e2e latency (us):
        avg latency: 50289
        min latency: 44639
        max latency: 62936
        p50 latency: 53648
        p90 latency: 54178
        p95 latency: 54212
        p99 latency: 54281
    model latency (us):
        avg latency: 50228
        min latency: 44564
        max latency: 62876
        p50 latency: 53608
        p90 latency: 54121
        p95 latency: 54152
        p99 latency: 54221
    ```
4. npz结果输出
    ```bash
    vamp -m deploy_weights/retinanet-int8-max-1_3_1024_1024-vacc/retinanet --vdsp_params ../vacc_code/vdsp_params/mmdet-retinanet_r50_fpn_1x_coco-vdsp_params.json -i 1 -b 1 -d 0 -p 1 --datalist datasets/coco_npz_datalist.txt --path_output npz_output
    ```
