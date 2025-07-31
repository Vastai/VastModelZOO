## Deploy

### step.1 获取预训练模型
- 模型来源
    ```bash
    - gitlab：git clone https://github.com/lyuwenyu/RT-DETR.git
    - commit: 5b628eaa0a2fc25bdafec7e6148d5296b144af85
    - model: https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    - config: https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml
    - weight: https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth
    ```

- 环境配置
    ```shell
    # torch版本不一致，可能导致onnx存在差异
    conda create -n detr python=3.8
    conda activate detr

    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip3 install numpy opencv-python onnx==1.14.0 onnxsim==0.4.36 onnxruntime==1.15.1 onnx_graphsurgeon==0.5.2 pycocotools scipy PyYAML tqdm decorator
    ```
    
- 源仓库提供了onnx导出脚本[rtdetr_pytorch/tools/export_onnx.py](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/tools/export_onnx.py)；此处需要进行适当修改以适配vacc推理
    - 删除后处理部分，不在onnx上实现；删除对应需要的'orig_target_sizes'输入，只保留一个'images'输入
    - 对应修改输出为'pred_logits'、'pred_boxes'
    - 修改后脚本：[export_onnx_nopost.py](./official/export_onnx_nopost.py)

    ```bash
    # 修改export_onnx_nopost.py开头的RT-DETR代码路径，需要从里面引用部分函数
    python tools/export_onnx_nopost.py \
    -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
    -r weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth \
    -f weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.onnx \
    --check --simplify 
    ```

    > 此步骤将自动从: https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth, 下载backbone得预训练权重至用户torch缓存路径: /home/user/.cache/torch/hub/checkpoints/

- 修改onnx文件，替换deform_attn、gridsampling等自定义算子，[export_custom.py](./official/export_custom.py)

    ```bash
    python export_custom.py \
    weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.onnx \
    weights/rtdetr_r18vd_dec3_6x_coco_from_paddle_custom.onnx
    ```

### step.2 获取数据集
- 依据原始仓库，使用[coco val2017](https://cocodataset.org/#download)验证模型精度
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)


### step.3 模型转换
- 注意当前只支持FP16的模型。
- 需要配置odsp环境变量
    ```
    cd /path/to/odsp_plugin/vastai/
    sudo ./build.sh
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/odsp_plugin/vastai/lib:/path/to/odsp_plugin/protobuf/lib/x86_64
    ```

1. 根据具体模型，修改编译配置
    - [official_rtdetr.yaml](../build_in/build/official_rtdetr.yaml)
        
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 模型编译
    ```bash
    cd rtdetr
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_rtdetr.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考: [rtdetr_vsx.py](../build_in/vsx/python/rtdetr_vsx.py)
    ```bash
    python ../build_in/vsx/python/rtdetr_vsx.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/official_rtdetr_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/rtdetr-vdsp_params.json \
        --label_txt /path/to/coco/coco.txt \
        --save_dir ./runstream_output \
        --device 0
    ```

    - 精度统计，参考：[eval_map.py](../../common/eval/eval_map.py)
    ```bash
        python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt ./runstream_output
    ```
    
    ```
    # fp16
    DONE (t=9.34s).
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.451
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.619
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.488
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.268
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.483
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.609
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.344
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.576
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.641
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.430
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.681
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.811
        {'bbox_mAP': 0.451, 'bbox_mAP_50': 0.619, 'bbox_mAP_75': 0.488, 'bbox_mAP_s': 0.268, 'bbox_mAP_m': 0.483, 'bbox_mAP_l': 0.609, 'bbox_mAP_copypaste': '0.451 0.619 0.488 0.268 0.483 0.609'}
    ```

### step.5 性能测试
    ```bash
    python3 ../build_in/vsx/python/rtdetr_prof.py \
        -m deploy_weights/official_rtdetr_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/rtdetr-vdsp_params.json  \
        --device_ids [0] \
        --batch_size 1 \
        --instance 1 \
        --iterations 100 \
        --shape "[3,640,640]" \
        --percentiles "[50,90,95,99]" \
        --input_host 1 \
        --queue_size 1
    ```

### Tips
<details><summary>基于RT-DETR仓库实现精度评估</summary>

 > **可选步骤**，基于RT-DETR仓库实现精度评估；与前文基于runstream脚本形式评估精度效果一致

    - 基于RT-DETR仓库实现精度评估，修改库内代码，使用vacc模型替换原始模型，批量推理可获得mAP精度信息
        - 首先，引用git patch修改，[rtdetr_modify.patch](./official/rtdetr_modify.patch)
        ```bash
        cd /path/to/RT-DETR
        git apply detection/rtdetr/source_code/official/rtdetr_modify.patch
        ```
        - 手动修改源仓库配置中的数据集路径和标签路径：[coco_detection.yml#L25](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/configs/dataset/coco_detection.yml#L25)
        - 执行批量精度测试脚本

        ```bash
        cd /path/to/RT-DETR/rtdetr_pytorch

        # run model
        python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml --test-only --resume weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth

        # run vacc
        python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml --test-only -resume vacc_deploy/rtdetr-fp16-none-1_3_640_640-vacc/mod --vacc
        ```

        ```
        # IoU metric: bbox

        # torch [1, 3, 640, 640]
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.637
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.503
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.284
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.497
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.629
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.363
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.616
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.687
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.497
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.733
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.858

        # vacc runmodel [1, 3, 640, 640]
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.637
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.502
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.282
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.497
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.629
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.363
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.616
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.687
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.497
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.734
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.858
        ```
</details>
