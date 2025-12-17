
## Build_In Deploy

### step.1 模型准备

1. 下载模型权重

    ```
    # awesome
    link: https://github.com/Tramac/awesome-semantic-segmentation-pytorch
    branch: master
    commit: b8366310de50869f89e836ed24de24edd432ece5
    ```

2. 模型导出

    一般在原始仓库内进行模型转为onnx或torchscript。在原仓库[demo.py#L48](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/scripts/demo.py#L48)或val脚本内，定义模型和加载训练权重后，添加以下脚本可实现：

    ```python
    # torch 1.8.0
    args.weights_test = "path/to/trained/weight.pth"
    model = self.model.eval()
    input_shape = (1, 3, 320, 320)
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    scripted_model.save(args.weights_test.replace(".pth", ".torchscript.pt"))
    scripted_model = torch.jit.load(args.weights_test.replace(".pth", ".torchscript.pt"))

    # onnx==10.0.0，opset 11
    import onnx
    torch.onnx.export(model, input_data, args.weights_test.replace(".pth", ".onnx"), input_names=["input"], output_names=["output"], opset_version=11)
    shape_dict = {"input": input_shape}
    onnx_model = onnx.load(args.weights_test.replace(".pth", ".onnx"))
    ```

### step.2 准备数据集
- 下载[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)数据集，解压

### step.3 模型转换
1. 根据具体模型修改模型转换配置文件
    - [awesome_fcn.yaml](../build_in/build/awesome_fcn.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd fcn
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/awesome_fcn.yaml
    ```

### step.4 模型推理

- 参考：[awesome_vsx_inference.py](../build_in/vsx/awesome_vsx_inference.py)
    ```bash
    python ../build_in/vsx/awesome_vsx_inference.py \
        --image_dir  /path/to/VOC2012/JPEGImages_val \
        --model_prefix_path deploy_weights/awesome_fcn_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/awesome-fcn8s_vgg16-vdsp_params.json \
        --mask_dir /path/to/SegmentationClass \
        --save_dir ./infer_output \
        --color_txt ../source_code/awesome/voc2012_colors.txt \
        --device 0
    ```

    ```
    # int8
    validation pixAcc: 87.792, mIoU: 50.596

    # fp16
    validation pixAcc: 87.769, mIoU: 50.513
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[awesome-fcn8s_vgg16-vdsp_params.json](../build_in/vdsp_params/awesome-fcn8s_vgg16-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/awesome_fcn_fp16/mod \
    --vdsp_params ../build_in/vdsp_params/awesome-fcn8s_vgg16-vdsp_params.json \
    -i 2 p 2 -b 1 -s [3,320,320]
    ```


2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；
    
    - 数据准备，基于[image2npz.py](../build_in/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`，注意只转换`VOC2012/ImageSets/Segmentation/val.txt`对应的验证集图像（配置相应路径）：
    ```bash
    python ../build_in/vdsp_params/image2npz.py \
        --dataset_path VOC2012/JPEGImages \
        --target_path  VOC2012/JPEGImages_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/awesome_fcn_fp16/mod \
    --vdsp_params ../build_in/vdsp_params/awesome-fcn8s_vgg16-vdsp_params.json \
    -i 2 p 2 -b 1 -s [3,320,320] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```

    - npz解析并统计精度，参考：[awesome_vamp_eval.py](../build_in/vdsp_params/awesome_vamp_eval.py)
    ```bash
    python ../build_in/vdsp_params/awesome_vamp_eval.py \
        --src_dir VOC2012/JPEGImages_val \
        --gt_dir VOC2012/SegmentationClass \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir ./npz_output \
        --input_shape 320 320 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```
