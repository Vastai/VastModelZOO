## Build_In Deploy

### step.1 模型准备

1. 下载模型权重

    ```
    # CoinCheung
    link: https://github.com/CoinCheung/BiSeNet
    branch: master
    commit: f2b901599752ce50656d2e50908acecd06f7eb47
    ```

2. 模型导出

    基于仓库脚本转换onnx和torchscript：[tools/export_onnx.py](https://github.com/CoinCheung/BiSeNet/blob/master/tools/export_onnx.py)

    ```python
    # https://github.com/CoinCheung/BiSeNet/blob/master/tools/export_onnx.py#L42
    # 修改配置参数和尺寸信息等，增加以下代码增加导出torchscript

    scripted_model = torch.jit.trace(net, dummy_input ).eval()
    scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))
    scripted_model = torch.jit.load(checkpoint.replace(".pth", ".torchscript.pt"))
    ```

    > Tips
    > 
    > 模型forward内存在argmax，当前提供的onnx已移除
    > 

### step.2 准备数据集
- 下载[cityscapes](https://www.cityscapes-dataset.com/)数据集，解压

### step.3 模型转换
1. 根据具体模型修改模型转换配置文件
    - [coincheung_config.yaml](../build_in/build/coincheung_config.yaml)

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    vamc compile ../vacc_code/build/coincheung_config.yaml
    ```

### step.4 模型推理
1. runstream推理，参考：[coincheung_vsx_inference.py](../build_in/vsx/coincheung_vsx_inference.py)，配置相关参数，执行进行runstream推理及获得精度指标
    ```bash
    python ../build_in/vsx/coincheung_vsx_inference.py \
        --image_dir path/to/cityscapes/leftImg8bit/val \
        --mask_dir path/to/cityscapes/gtFine/val \
        --model_prefix_path deploy_weights/coincheung_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/coincheung-bisenetv1-vdsp_params.json \
        --label_txt ../build_in/runmodel/cityscapes_colors.txt \
        --save_dir ./runstream_output
    ```
    - 注意替换命令行中--image_dir和--mask_dir为实际路径


### step.5 性能精度测试
1. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path cityscapes/leftImg8bit/val \
    --target_path  cityscapes/leftImg8bit/val_npz \
    --text_path npz_datalist.txt
    ```
2. 性能测试，配置vdsp参数[coincheung-bisenetv2-vdsp_params.json](../build_in/vdsp_params/coincheung-bisenetv2-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/bisenetv2-int8-kl_divergence-3_736_960-vacc/bisenetv2 \
    --vdsp_params ../build_in/vdsp_params/coincheung-bisenetv2-vdsp_params.json \
    -i 1 p 1 -b 1
    ```

> 可选步骤，和step.4内使用runstream脚本方式的精度测试基本一致

3. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/bisenetv2-int8-kl_divergence-3_736_960-vacc/bisenetv2 \
    --vdsp_params build_in/vdsp_params/coincheung-bisenetv2-vdsp_params.json \
    -i 1 p 1 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
4. [coincheung_vamp_eval.py](../build_in/vdsp_params/coincheung_vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../build_in/vdsp_params/coincheung_vamp_eval.py \
    --src_dir cityscapes/leftImg8bit/val \
    --gt_dir cityscapes/gtFine/val \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 736 960 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```
- 
    <details><summary>eval metrics</summary>

    ```
    bisenetv1_city_new.pth
    validation pixAcc: 94.125, mIoU: 67.629

    bisenetv1_city-fp16-none-3_736_960-vacc
    validation pixAcc: 94.099, mIoU: 67.457

    bisenetv1_city-int8-kl_divergence-3_736_960-vacc
    validation pixAcc: 93.604, mIoU: 64.615

    bisenetv2_city.pth
    validation pixAcc: 94.713, mIoU: 69.778

    bisenetv2_city-fp16-none-3_736_960-vacc
    validation pixAcc: 94.719, mIoU: 69.769

    bisenetv2_city-int8-kl_divergence-3_736_960-vacc
    validation pixAcc: 94.503, mIoU: 68.111
    ```
    </details>
