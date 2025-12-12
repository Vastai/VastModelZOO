
## Build_In Deploy

### step.1 模型准备

1. 下载模型权重

    ```
    link: https://github.com/zllrunning/face-parsing.PyTorch
    branch: master
    commit: d2e684cf1588b46145635e8fe7bcc29544e5537e
    ```

2. 模型导出

    一般在原始仓库内进行模型转为onnx或torchscript。在原仓库test或val脚本内，如[test.py](./face_parsing/test.py)，定义模型和加载训练权重后，添加以下脚本可实现：

    ```python
    checkpoint = save_pth

    input_shape = (1, 3, 512, 512)
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)

    scripted_model = torch.jit.trace(net, input_data).eval()
    scripted_model.save(checkpoint.replace(".pth", "-512.torchscript.pt"))
    scripted_model = torch.jit.load(checkpoint.replace(".pth", "-512.torchscript.pt"))

    # # onnx==10.0.0，opset 10
    # import onnx
    # torch.onnx.export(net, input_data, checkpoint.replace(".pth", "-512.onnx"), input_names=["input"], output_names=["output"], opset_version=11)
    # shape_dict = {"input": input_shape}
    # onnx_model = onnx.load(checkpoint.replace(".pth", "-512.onnx"))
    ```

### step.2 准备数据集
- 下载[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)数据集，解压
- 这个仓库的标签和数据集官方标签顺序不一样，需要用仓库脚本[prepropess_data.py](./face_parsing/prepropess_data.py)（参考自[源仓库](https://github.com/zllrunning/face-parsing.PyTorch/blob/master/prepropess_data.py)生成验证数据集）


### step.3 模型转换
1. 根据具体模型修改模型转换配置文件
    - [zllrunning_config.yaml](../build_in/build/zllrunning_config.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd bisenet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/zllrunning_config.yaml
    ```

### step.4 模型推理
1. 参考[zllrunning_vsx_inference.py](../build_in/vsx/zllrunning_vsx_inference.py)，配置相关参数，进行推理及获得精度指标
    ```bash
    python ../build_in/vsx/zllrunning_vsx_inference.py \
        --image_dir /path/to/CelebAMask-HQ/bisegnet_test_img \
        --mask_dir /path/to/CelebAMask-HQ/bisegnet_test_mask \
        --model_prefix_path deploy_weights/bisenet_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/zllrunning-bisenet-vdsp_params.json \
        --save_dir ./infer_output
    ```
    - 注意替换命令行中--image_dir和--mask_dir为实际路径


### step.5 性能精度测试
1. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path CelebAMask-HQ/bisegnet_test_img \
    --target_path  CelebAMask-HQ/bisegnet_test_img_npz \
    --text_path npz_datalist.txt
    ```

2. 性能测试，配置vdsp参数[zllrunning-bisenet-vdsp_params.json](../build_in/vdsp_params/zllrunning-bisenet-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/bisenet-int8-kl_divergence-3_512_512-vacc/bisenet \
    --vdsp_params ../build_in/vdsp_params/zllrunning-bisenet-vdsp_params.json \
    -i 2 p 2 -b 1
    ```

> 可选步骤，和step.4的精度测试基本一致

3. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/bisenet-int8-kl_divergence-3_512_512-vacc/bisenet \
    --vdsp_params build_in/vdsp_params/zllrunning-bisenet-vdsp_params.json \
    -i 2 p 2 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
4. [zllrunning_vamp_eval.py](../build_in/vdsp_params/zllrunning_vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../build_in/vdsp_params/zllrunning_vamp_eval.py \
    --src_dir CelebAMask-HQ/bisegnet_test_img \
    --gt_dir CelebAMask-HQ/bisegnet_test_mask \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 512 512 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```