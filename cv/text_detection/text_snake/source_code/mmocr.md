## mmocr

```
# mmocr
link: https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/textsnake
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
    python tools/deploy.py configs/mmocr/text-detection/text-detection_onnxruntime_static.py ../mmocr/configs/textdet/textsnake/textsnake_resnet50_fpn-unet_1200e_ctw1500.py ../mmocr/models/textsnake/textsnake_resnet50_fpn-unet_1200e_ctw1500_20220825_221459-c0b6adc4.pth demo/resources/face.png --work-dir mmdeploy_models/textsnake_resnet50_fpn --device cpu --dump-info
    ```


### step.2 准备数据集
- 下载[CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)数据集，通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [mmocr_text_snake.yaml](../build_in/build/mmocr_text_snake.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd text_snake
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/mmocr_text_snake.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考：[text_snake_vsx.py](../build_in/vsx/python/text_snake_vsx.py)
    ```bash
    python ../build_in/vsx/python/text_snake_vsx.py \
        --file_path  /path/to/ctw1500/test/text_image/  \
        --model_prefix_path deploy_weights/mmocr_text_snake_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/mmocr-textsnake_resnet50_fpn-vdsp_params.json \
        --save_dir ./runstream_output \
        --device 0
    ```

    - 参考[script.py](../source_code/eval/script.py)，对上述结果压缩后进行预测
    ```
    cd runstream_output
    zip -r vsx_pred.zip *
    mv vsx_pred.zip ../../source_code/eval/
    cd ../../source_code/eval/
    python script.py -g=ctw1500-gt.zip -s=vsx_pred.zip
    ```

    ```
    # fp16
    num_gt, num_det:  3068 3043
    Origin:
    recall:  0.7754 precision:  0.7818 hmean:  0.7786
    TIoU-metric:
    tiouRecall: 0.56 tiouPrecision: 0.0 tiouHmean: 0.0

    # int8
    num_gt, num_det:  3068 3038
    Origin:
    recall:  0.781 precision:  0.7887 hmean:  0.7848
    TIoU-metric:
    tiouRecall: 0.564 tiouPrecision: 0.0 tiouHmean: 0.0
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[mmocr-textsnake_resnet50_fpn-vdsp_params.json](../build_in/vdsp_params/mmocr-textsnake_resnet50_fpn-vdsp_params.json)，执行：
    ```bash
    vamp -m ./deploy_weights/mmocr_text_snake_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/mmocr-textsnake_resnet50_fpn-vdsp_params.json -i 1 p 1 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py --dataset_path /path/to/ocr/ctw1500/test_images/ --target_path /path/to/ocr/ctw1500/test_images_npz --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m ./deploy_weights/mmocr_text_snake_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/mmocr-textsnake_resnet50_fpn-vdsp_params.json -i 1 p 1 -b 1 --datalist npz_datalist.txt --path_output npz_output
    ```

    - 解析npz结果，参考：[npz_decode.py](../build_in/vdsp_params/npz_decode.py)，
    ```bash
    python ../build_in/vdsp_params/npz_decode.py \
        --gt_dir /path/to/ocr/ctw1500/test_images/ \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output 
    ```

    - 统计精度结果
    ```bash
    cd npz_output
    zip -r vsx_int8_pred.zip *
    mv vsx_int8_pred.zip ../../source_code/eval/
    cd ../../source_code/eval/
    python script.py -g=ctw1500-gt.zip -s=vsx_int8_pred.zip
    ```
