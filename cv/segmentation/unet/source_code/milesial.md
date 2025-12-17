## Build_In Deploy

### step.1 模型准备
1. 下载模型权重

    ```
    link：https://github.com/milesial/Pytorch-UNet
    branch: master
    commit: a013c80ca6b011ba34ba0700b961b77e0ed003a2
    ```

2. 模型导出

根据原始仓库即可进模型导出：
    - 在[predict.py#L24](https://github.com/milesial/Pytorch-UNet/blob/master/predict.py#L24)，增加torchscript和onnx转换代码：

    ```python
    dump_input = torch.rand((1, 3, 512, 512))
    export_onnx_file = args.model.replace(".pth",".onnx")
    torch.onnx.export(net.cpu(), dump_input.cpu(), export_onnx_file, verbose=True, opset_version=11, input_names=["input"])

    traced_script = torch.jit.trace(net, dump_input)
    traced_script.save(args.model.replace(".pth",".torchscript.pt"))
    ```

### step.2 准备数据集
- 下载[carvana](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)数据集，解压

### step.3 模型转换
1. 根据具体模型修改模型转换配置文件
    - [milesial_unet.yaml](../build_in/build/milesial_unet.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd unet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/milesial_unet.yaml
    ```

### step.4 模型推理
1. 参考：[milesial_vsx.py](../build_in/vsx/python/milesial_vsx.py)，修改参数并运行如下脚本
    ```bash
    python ../build_in/vsx/python/milesial_vsx.py \
        --file_path  /path/to/carvana/imgs \
        --model_prefix_path deploy_weights/milesial_unet_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/milesial-unet_scale0.5-vdsp_params.json \
        --gt_path /path/to/carvana/masks \
        --save_dir ./infer_output \
        --device 0
    ```

### step.5 性能精度测试
1. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path carvana/imgs \
    --target_path  carvana/imgs_npz \
    --text_path npz_datalist.txt
    ```

2. 性能测试，配置vdsp参数[milesial-unet_scale0.5-vdsp_params.json](../build_in/vdsp_params/milesial-unet_scale0.5-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/milesial_unet_int8/mod \
    --vdsp_params ../build_in/vdsp_params/milesial-unet_scale0.5-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,512,512]
    ```
    
> 可选步骤，和step.4的精度测试基本一致

3. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/milesial_unet_int8/mod \
    --vdsp_params ../build_in/vdsp_params/milesial-unet_scale0.5-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,512,512] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```

4. [milesial-vamp_eval.py](../build_in/vdsp_params/milesial-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../build_in/vdsp_params/milesial-vamp_eval.py \
    --src_dir carvana/imgs \
    --gt_dir carvana/masks \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 512 512 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```