# mlcommons

### step.1 获取预训练模型

- 转换依赖原始training仓库代码
    ```bash
    git clone https://github.com/mlcommons/training.git

    # 日志系统
    git clone https://github.com/mlperf/logging.git mlperf-logging
    pip install -e mlperf-logging
    ```
- 由于vacc不支持Retinanet的后处理，只能对forward部分截断处理（backbone+neck+head），输出10个featuremap，分类和回归分支各5个
- 修改模型forward代码：拷贝脚本[retinanet_forward.py](./mlcommons/retinanet_forward.py)至{training/single_stage_detector/ssd/model}目录

- 模型导出：拷贝脚本[onnx_convert.py](./mlcommons/onnx_convert.py)至{training/single_stage_detector/ssd}目录

  ```bash
  # onnx export
  python onnx_convert.py --weights retinanet_model_10.pth --output resnext50_32x4d_fpn_forward.onnx
  
  # onnxsim
  onnxsim resnext50_32x4d_fpn_forward.onnx resnext50_32x4d_fpn_forward_sim.onnx
  ```
  
### step.2 准备数据集
- [评估数据集](https://storage.googleapis.com/openimages/web/download_v6.html)
- [校准数据集]：同上，随机选取100张
- [gt: openimages-mlperf.json](https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/README.md#datasets)


### step.3 模型转换

1. 根据具体模型修改模型转换配置文件[mlcommons_config.yaml](../build_in/build/mlcommons_retinanet.yaml)：

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd retinanet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/mlcommons_retinanet.yaml
    ```

### step.4 模型推理

> `engine.type: vacc`
> 
> **vsx**形式

1. 可以利用[脚本](../build_in/vsx/python/mlcommons_retinanet_vsx.py)生成预测的txt结果

    ```
    python ../build_in/vsx/python/mlcommons_retinanet_vsx.py \
    --image_dir openimages/validation/data \
    --model_prefix_path ./deploy_weights/resnext50_32x4d_fpn_forward-int8-max-1_3_800_800-vacc/mod \
    --vdsp_params_info ../build_in/vdsp_params/mlcommons-resnext50_32x4d_fpn-vdsp_params.json \
    --gt_file openimages/annotations/openimages-mlperf.json \
    --save_dir ./vsx_vacc_results \
    --draw_image
    ```

2. [eval_map.py](./mlcommons/eval_map.py)，精度统计，指定`openimages-mlperf.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../source_code/mlcommons/eval_map.py --gt openimages-mlperf.json --txt ./vsx_vacc_results
   ```

    > Tips
    >
    > 阈值等参数设置，使用默认：https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/model/retinanet.py#L329
    > 
    >  score_thresh=0.05, nms_thresh=0.5, detections_per_img=300

    - <details><summary>精度</summary>

            '''
            [1, 3, 800, 800]
            onnx
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.385
            Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.534
            Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.418
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.037
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.131
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.427
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.426
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.603
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.629
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.096
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.342
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.678
            {'bbox_mAP': 0.385, 'bbox_mAP_50': 0.534, 'bbox_mAP_75': 0.418, 'bbox_mAP_s': 0.037, 'bbox_mAP_m': 0.131, 'bbox_mAP_l': 0.427, 'bbox_mAP_copypaste': '0.385 0.534 0.418 0.037 0.131 0.427'}

            vsx_vacc_fp16 BILINEAR_PILLOW
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.385
            Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.534
            Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.416
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.426
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.427
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.603
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.629
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.104
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.334
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.678
            {'bbox_mAP': 0.385, 'bbox_mAP_50': 0.534, 'bbox_mAP_75': 0.416, 'bbox_mAP_s': 0.039, 'bbox_mAP_m': 0.128, 'bbox_mAP_l': 0.426, 'bbox_mAP_copypaste': '0.385 0.534 0.416 0.039 0.128 0.426'}

            vsx_vacc_int8-kl_divergence BILINEAR_PILLOW
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.384
            Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.531
            Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.416
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.037
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.425
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.428
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.602
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.628
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.101
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.331
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.677
            {'bbox_mAP': 0.384, 'bbox_mAP_50': 0.531, 'bbox_mAP_75': 0.416, 'bbox_mAP_s': 0.037, 'bbox_mAP_m': 0.128, 'bbox_mAP_l': 0.425, 'bbox_mAP_copypaste': '0.384 0.531 0.416 0.037 0.128 0.425'}
        '''
    </details>


### step.5 性能测试
1. 性能测试
    ```bash
    export VSX_DISABLE_DEEPBIND=1
    vamp -m deploy_weights/resnext50_32x4d_fpn_forward_tvmlib-int8-kl_divergence-1_3_800_800-vacc/mod \
        --vdsp_params ../build_in/vdsp_params/mlcommons-resnext50_32x4d_fpn-vdsp_params.json -i 1 p 1 -b 1 -d 0
    ```
   
### Tips
- VDSP内的ResizeType参数选择`BILINEAR_PILLOW`，精度最佳
- VDSP不支持此模型后处理，所以后处理在torch端实现