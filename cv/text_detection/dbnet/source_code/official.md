## PPOCR

```
link: https://github.com/MhLiao/DB
branch: master
commit: e5a12f5c2f0c2b4a345b5b8392307ef73481d5f6
```

### step.1 模型准备
通过[convert_to_onnx.py](../source_code/official/convert_to_onnx.py)，将pth模型转为onnx或torchscript。

### step.2 准备数据集
- 下载[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)数据集
  - 测试图像：[ch4_test_images](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=4&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy8/Y29tPWRvd25sb2FkcyZhY3Rpb249ZG93bmxvYWQmZmlsZT1jaDRfdGVzdF9pbWFnZXMuemlw)
  - 测试图像标签：[test_icdar2015_label.txt](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=4&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9kb3dubG9hZHMvQ2hhbGxlbmdlNF9UZXN0X1Rhc2sxX0dULnppcA==)
    - 需要将下载的官网`label`转换支持的数据格式`test_icdar2015_label.txt`,具体转换方式可参考[测评数据集说明](../README.md)
  - 通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件

### step.3 模型转换
1. 根据具体模型修改配置文件
    -[official_dbnet.yaml](../build_in/build/official_dbnet.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd dbnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_config.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights/official_dbnet_run_stream_int8`文件夹，其中包含转换后的模型文件。

### step.4 模型推理
1. 参考[dbnet_vsx.py](../build_in/vsx/python/dbnet_vsx.py)，执行下面命令
    ```bash
    python ../build_in/vsx/python/dbnet_vsx.py \
        --file_path path/to/icdar2015/Challenge4/ch4_test_images \
        --model_prefix_path deploy_weights/official_dbnet_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-dbnet_resnet18-vdsp_params.json \
        --label_txt path/to/icdar2015/Challenge4/test_icdar2015_label.txt
    ```
    - 注意替换命令行中--file_path和--label_txt为实际路径
    - 精度信息就在打印信息中，如下：
    ```
    metric:  {'precision': 0.7871376811594203, 'recall': 0.4183919114106885, 'hmean': 0.5463690663313423}
    ```

### step.5 性能精度测试
1. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path lane_detection/Tusimple/clips \
    --target_path lane_detection/Tusimple/clips_npz \
    --text_path npz_datalist.txt
    ```

2. 性能测试，配置vdsp参数[official-dbnet_resnet18-vdsp_params.json](../build_in/vdsp_params/official-dbnet_resnet18-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/official_dbnet_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/official-dbnet_resnet18-vdsp_params.json -i 1 p 1 -b 1 -s [3,640,640]
    ```

3. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_dbnet_run_stream_int8/mod \
    --vdsp_params ../build_in/vdsp_params/official-dbnet_resnet18-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,640,640] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
    
4. [vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
    --gt_dir icdar2015/Challenge4/ch4_test_images \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir npz_output \
    --input_shape 640 640 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```