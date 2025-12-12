## mmocr

```
# mmocr
link: https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/fcenet
branch: main
commit: b18a09b2f063911a2de70f477aa21da255ff505d
```

### step.1 获取预训练模型

1. clone mmocr、mmdeploy库，并安装mmcv、mmocr、mmdeploy等环境依赖
2. 下载相应pth模型文件
3. mmdeploy转换时默认尺寸为`736x736`，可以通过修改config中`test_pipeline`设置模型尺寸(1280X736)，如下：
    ```python
    test_pipeline = [
      dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
      dict(type='Resize', scale=(1280, 736), keep_ratio=False),
      # add loading annotation after ``Resize`` because ground truth
      # does not need to do resize data transform
      dict(
         type='LoadOCRAnnotations',
         with_polygon=True,
         with_bbox=True,
         with_label=True),
      dict(
         type='PackTextDetInputs',
         meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
   ]
    ```

4. 通过mmdeploy库转出onnx文件，命令如下

    ```bash
    python tools/deploy.py configs/mmocr/text-detection/text-detection_onnxruntime_static.py ../mmocr/configs/textdet/fcenet/fcenet_resnet50_fpn_1500e_icdar2015.py ../mmocr/models/fcenet/fcenet_resnet50_fpn_1500e_icdar2015_20220826_140941-167d9042.pth demo/resources/face.png --work-dir mmdeploy_models/fcenet_resnet50_fpn_1500e_icdar2015  --device cpu --dump-info
    ```


### step.2 准备数据集
- 下载[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)数据集，通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [mmocr_fcenet.yaml](../build_in/build/mmocr_fcenet.yaml)
    

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd fcenet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/mmocr_fcenet.yaml
    ```

### step.4 模型推理

- 参考：[fcenet_vsx.py](../build_in/vsx/python/fcenet_vsx.py)
    ```bash
    python ../build_in/vsx/python/fcenet_vsx.py \
        --file_path  /path/to/ch4_test_images  \
        --model_prefix_path deploy_weights/mmocr_fcenet_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/mmocr-fcenet_resnet50_oclip_fpn_1500e_icdar2015-vdsp_params.json \
        --label_txt /path/to/test_icdar2015_label.txt \
        --device 0
    ```

    ```
    # fp16
    metric:  {'precision': 0.7688249400479616, 'recall': 0.7717862301396244, 'hmean': 0.770302739067756}

    # int8
    metric:  {'precision': 0.7830791933103788, 'recall': 0.7664901299951854, 'hmean': 0.7746958637469588}
    ```


### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[mmocr-fcenet_resnet50_oclip_fpn_1500e_icdar2015-vdsp_params.json](../build_in/vdsp_params/mmocr-fcenet_resnet50_oclip_fpn_1500e_icdar2015-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/mmocr_fcenet_int8/mod --vdsp_params ../build_in/vdsp_params/mmocr-fcenet_resnet50_oclip_fpn_1500e_icdar2015-vdsp_params.json -i 1 -p 1 -d 0 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path /path/to/ch4_test_images \
        --target_path /path/to/ch4_test_images_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/mmocr_fcenet_int8/mod --vdsp_params ../build_in/vdsp_params/mmocr-fcenet_resnet50_oclip_fpn_1500e_icdar2015-vdsp_params.json -i 1 -p 1 -d 0 -b 1 --datalist npz_datalist.txt --path_output output
    ```

    - 解析npz结果，参考：[npz_decode.py](../build_in/vdsp_params/npz_decode.py)
    ```bash
    python ../build_in/vdsp_params/npz_decode.py --gt_dir /path/to/ocr/ICDAR2015/ch4_test_images --input_npz_path npz_datalist.txt --out_npz_dir output
    ```