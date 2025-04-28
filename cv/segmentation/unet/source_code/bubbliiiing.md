## Build_In Deploy

### step.1 模型准备

1. 下载模型权重

    ```
    link：https://github.com/bubbliiiing/unet-pytorch
    branch: main
    commit: 8ab373232c5c3a1877f3e84e6f5d97404089c20f
    ```

2. 模型导出

    根据原始仓库即可进模型导出：
    - 将[predict.py#L26](https://github.com/bubbliiiing/unet-pytorch/blob/main/predict.py#L26)修改为`export_onnx`模式
    - 在[unet.py#L267](https://github.com/bubbliiiing/unet-pytorch/blob/main/unet.py#L267)，增加torchscript转换代码：
    ```python
    scripted_model = torch.jit.trace(self.net, im).eval()
    torch.jit.save(scripted_model, model_path.replace(".onnx", ".torchscript.pt"))
    ```
    - 执行[predict.py](https://github.com/bubbliiiing/unet-pytorch/blob/main/predict.py#L26)即可导出onnx和torchscript（opset_version=11）

    > Tips
    >
    > opset_version=10时转onnx报错：RuntimeError: ONNX export failed: Couldn't export operator aten::upsample_bilinear2d


### step.2 准备数据集
- 下载[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)数据集


### step.3 模型转换
1. 参考瀚博训推软件生态链文档，获取模型转换工具: [vamc v3.0+](../../../../docs/vastai_software.md)
2. 根据具体模型修改模型转换配置文件
    - [bubbliiiing_unet.yaml](../build_in/build/bubbliiiing_unet.yaml)
3. 执行转换
    ```bash
    vamc compile ../build_in/build/bubbliiiing_unet.yaml
    ```


### step.4 模型推理
1. 参考瀚博训推软件生态链文档，获取模型推理工具：[vaststreamx v2.8+](../../../../docs/vastai_software.md)
2. runstream推理，参考：[bubbliiiing_vsx.py](../build_in/vsx/python/bubbliiiing_vsx.py)，修改参数并运行如下脚本
    ```bash
    python ../build_in/vsx/python/bubbliiiing_vsx.py \
        --file_path  /path/to/VOC2012/JPEGImages_val \
        --model_prefix_path deploy_weights/bubbliiiing_unet_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/bubbliiiing-unet_resnet50-vdsp_params.json \
        --gt_path /path/to/SegmentationClass \
        --save_dir ./runstream_output \
        --device 0
    ```


### step.5 性能精度
1. 参考瀚博训推软件生态链文档，获取模型性能测试工具：[vamp v2.4+](../../../../docs/vastai_software.md)

2. 基于[image2npz.py](../build_in/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`，注意只转换`VOC2012/ImageSets/Segmentation/val.txt`对应的验证集图像（配置相应路径）：
    ```bash
    python ../build_in/vdsp_params/image2npz.py \
    --dataset_path VOC2012/JPEGImages \
    --target_path  VOC2012/JPEGImages_npz \
    --text_path npz_datalist.txt
    ```

3. 性能测试，配置vdsp参数[bubbliiiing-unet_resnet50-vdsp_params.json](../build_in/vdsp_params/bubbliiiing-unet_resnet50-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/bubbliiiing_unet_run_stream_int8/mod \
    --vdsp_params ../build_in/vdsp_params/bubbliiiing-unet_resnet50-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256]
    ```
    
> 可选步骤，和step.4内使用runstream脚本方式的精度测试基本一致

4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/unet_resnet50-int8-kl_divergence-3_256_256-vacc/unet_resnet50 \
    --vdsp_params build_in/vdsp_params/bubbliiiing-unet_resnet50-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```

5. [bubbliiiing-vamp_eval.py](../build_in/vdsp_params/bubbliiiing-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../build_in/vdsp_params/bubbliiiing-vamp_eval.py \
    --src_dir VOC2012/JPEGImages_val \
    --gt_dir VOC2012/SegmentationClass \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 256 256 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```

