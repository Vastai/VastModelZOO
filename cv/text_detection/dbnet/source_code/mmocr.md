## MMOCR

```
link: https://github.com/open-mmlab/mmocr/blob/main/configs/backbone/oclip/README.md
branch: v1.0.1
commit: b18a09b2f063911a2de70f477aa21da255ff505d
```

### step.1 模型准备

1. clone mmocr、mmdeploy库，并安装mmcv、mmocr、mmdeploy等环境依赖
2. 下载相应pth模型文件
3. 修改`mmocr/models/textdet/heads/db_head.py`文件中`DBHead`类`forward`函数，如下：
    ```python
    def forward(self,
                img: Tensor,
                data_samples: Optional[List[TextDetDataSample]] = None,
                mode: str = 'predict') -> Tuple[Tensor, Tensor, Tensor]:
        prob_logits = self.binarize(img)#.squeeze(1)
        # prob_map = self.sigmoid(prob_logits)
        if mode == 'predict':
            return prob_logits
    ```

4. mmdeploy转换时默认尺寸为`736x736`，可以通过修改config中`test_pipeline`设置模型尺寸(1280X736)，如下：
    ```python
    test_pipeline = [
        dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
        dict(type='Resize', scale=(1280, 736), keep_ratio=False),
        dict(
            type='LoadOCRAnnotations',
            with_polygon=True,
            with_bbox=True,
            with_label=True,
        ),
        dict(
            type='PackTextDetInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ]

    ```

5. 通过mmdeploy库转出onnx文件，命令如下

    ```bash
    python tools/deploy.py configs/mmocr/text-detection/text-detection_onnxruntime_static.py ../mmocr/configs/textdet/dbnet/dbnet_resnet50-oclip_1200e_icdar2015.py ../mmocr/models/dbnet/dbnet_resnet50-oclip_1200e_icdar2015_20221102_115917-bde8c87a.pth demo/resources/face.png --work-dir mmdeploy_models/dbnet_resnet50-oclip_1200e_icdar2015 --device cpu --dump-info
    ```

### step.2 准备数据集
- 下载[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)数据集
  - 测试图像：[ch4_test_images](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=4&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy8/Y29tPWRvd25sb2FkcyZhY3Rpb249ZG93bmxvYWQmZmlsZT1jaDRfdGVzdF9pbWFnZXMuemlw)
  - 测试图像标签：[test_icdar2015_label.txt](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=4&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9kb3dubG9hZHMvQ2hhbGxlbmdlNF9UZXN0X1Rhc2sxX0dULnppcA==)
    - 需要将下载的官网`label`转换支持的数据格式`test_icdar2015_label.txt`,具体转换方式可参考[测评数据集说明](../README.md)
  - 通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件

### step.3 模型转换
1. 根据具体模型修改配置文件
    - [mmocr_dbnet.yaml](../build_in/build/mmocr_dbnet.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd dbnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/mmocr_dbnet.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights/mmocr_dbnet_run_stream_int8`文件夹，其中包含转换后的模型文件。

### step.4 模型推理
1. 参考[mmocr_dbnet_vsx.py](../build_in/vsx/python/mmocr_dbnet_vsx.py)，进行vsx推理和eval评估
    ```bash
    python ../build_in/vsx/python/mmocr_dbnet_vsx.py  \
        --file_path path/to/icdar2015/Challenge4/ch4_test_images \
        --model_prefix_path deploy_weights/mmocr_dbnet_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/mmocr-dbnet_resnet18_fpnc_1200e_icdar2015-vdsp_params.json \
        --label_txt path/to/icdar2015/Challenge4/test_icdar2015_label.txt
    ```
    - 注意替换命令行中--file_path和--label_txt为实际路径
    - 精度信息就在打印信息中，结果如下：
    ```
    metric:  {'precision': 0.8771138669673055, 'recall': 0.7491574386133847, 'hmean': 0.8081017917424047}
    ```

### step.5 性能精度测试
1. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path ch4_test_images \
    --target_path ch4_test_images_npz \
    --text_path npz_datalist.txt
    ```
2. 性能测试，配置vdsp参数[mmocr-dbnet_resnet18_fpnc_1200e_icdar2015-vdsp_params.json](../build_in/vdsp_params/mmocr-dbnet_resnet18_fpnc_1200e_icdar2015-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/mmocr_dbnet_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/mmocr-dbnet_resnet18_fpnc_1200e_icdar2015-vdsp_params.json  -i 1 p 1 -b 1
    ```

3. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/mmocr_dbnet_run_stream_int8/mod \
    --vdsp_params ../build_in/vdsp_params/mmocr-dbnet_resnet18_fpnc_1200e_icdar2015-vdsp_params.json \
    -i 1 p 1 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```

4. [mmocr_vamp_eval.py](../build_in/vdsp_params/mmocr_vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../build_in/vdsp_params/mmocr_vamp_eval.py \
    --gt_dir icdar2015/Challenge4/ch4_test_images \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir npz_output \
    --input_shape 736 1280 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```
