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
- 准备[COCO](https://cocodataset.org/#download)数据集


### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[mmdet_yolof.yaml](../vacc_code/build/mmdet_yolof.yaml)：
    ```bash
    vamc build ../vacc_code/build/mmdet_yolof.yaml
    ```

### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    ```bash
    vamp -m deploy_weights/yolof-int8-kl_divergence-1_3_1280_1280-vacc/yolof --vdsp_params vdsp_params.json -i 1 -b 1 -d 0 -p 1
    ```

    ```bash
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    ai utilize (%): 94.7627
    temperature (°C): 40.1822
    card power (W): 45.2899
    die memory used (MB): 1452.95
    throughput (qps): 156.071
    e2e latency (us):
        avg latency: 15781
        min latency: 12527
        max latency: 24866
        p50 latency: 18484
        p90 latency: 18898
        p95 latency: 18928
        p99 latency: 18976
    model latency (us):
        avg latency: 15741
        min latency: 12516
        max latency: 24812
        p50 latency: 18425
        p90 latency: 18852
        p95 latency: 18882
        p99 latency: 18924
    ```
4. npz结果输出
    ```bash
    vamp -m deploy_weights/yolof-int8-kl_divergence-1_3_1280_1280-vacc/yolof --vdsp_params vdsp_params.json -i 1 -b 1 -d 0 -p 1 --datalist datasets/coco_npz_datalist.txt --path_output npz_output
    ```
5. [vamp_decode.py](../vacc_code/vdsp_params/vamp_decode.py)，解析vamp输出的npz文件，进行绘图和保存txt结果
    ```bash
    python npz_decode.py --label_txt ../../eval/coco.txt --input_image_dir ~/datasets/detection/coco_val2017 --vamp_datalist_path ~/datasets/detection/datalist_npz.txt --vamp_output_dir npz_output
    ```
6. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt path/to/vamp_draw_output
   ```
