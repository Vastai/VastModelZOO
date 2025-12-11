## Deploy
### step.1 获取模型
- 原始仓库
    ```bash
    git clone https://github.com/microsoft/Swin-Transformer.git
    
    # branch
    main
    # commit
    968e6b5
    ```
- 原始权重链接：[swin_base_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth)

- onnx导出
    > 以下步骤环境依赖参考Tips
1. 基于源repo导出onnx， 详见[export_onnx.py](./export_onnx.py)
    ```
    git clone https://github.com/microsoft/Swin-Transformer.git
    mv source_code/export_onnx.py Swin-Transformer/models
    cd Swin-Transformer/models
    python export_onnx.py
    ```
2. 替换custom op，详见[convert_custom_op.py](./convert_custom_op.py)
    ```bash
    python convert_custom_op.py
    ```

### step.2 获取数据集
- [校准数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../common/label/imagenet.txt)
- [label_dict](../../common/label/imagenet1000_clsid_to_human.txt)

### step.3 环境准备
    ```bash
    # 需要配置odsp环境变量
    cd /path/to/odsp_plugin/vastai/
    sh build.sh
    export LD_LIBRARY_PATH=/path/to/odsp_plugin/vastai/odsp_plugin/protobuf/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/path/to/odsp_plugin/vastai/odsp_plugin/vastai/lib:$LD_LIBRARY_PATH
    ```

### step.4 模型转换
1. 根据具体模型，修改编译配置
    - [microsoft_swin_transformer.yaml](../build_in/build/microsoft_swin_transformer.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 模型编译

    ```bash
    cd swin_transformer
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/microsoft_swin_transformer.yaml
    ```

### step.5 模型推理
 - 参考：[infer_swin.py](../build_in/vsx/infer_swin.py)

    准备测试前，需先准备数据集路径文件`ILSVRC2012_img_val_filelist.txt`，格式如下：

    ```
    ......
    ILSVRC2012_img_val/n12267677/ILSVRC2012_val_00001058.JPEG
    ILSVRC2012_img_val/n12267677/ILSVRC2012_val_00001117.JPEG
    ILSVRC2012_img_val/n12267677/ILSVRC2012_val_00002139.JPEG
    ......
    ```

    运行如下命令进行精度测试：

    ```bash
    python ../build_in/vsx/infer_swin.py \
        -m deploy_weights/microsoft_swin_transformer_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/microsoft-swin_b-vdsp_params.json \
        --device_id 0 \
        --label_file /path/to/imagenet.txt \
        --dataset_filelist /path/to/ILSVRC2012_img_val_filelist.txt \
        --dataset_root /path/to/ImageNet \
        --dataset_output_file result.txt
    ```

    - 精度评估
    ```
    python ../../common/eval/eval_topk.py result.txt
    ```

    ```
    # fp16
    [VACC]:  top1_rate: 82.988 top5_rate: 96.296
    ```


### step.6 性能测试
1. 使用[infer_swin_prof.py](../build_in/vsx/infer_swin_prof.py)进行benchmark
    ```bash
    python3 ../build_in/vsx/infer_swin_prof.py \
        -m deploy_weights/microsoft_swin_transformer_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/microsoft-swin_b-vdsp_params.json \
        --device_ids [0] \
        --batch_size 1 \
        --instance 1 \
        --iterations 300 \
        --shape [3,224,224] \
        --percentiles [50,90,95,99] \
        --input_host 1 \
        --queue_size 1

    ```

### Tips
- 仅支持FP16
- 环境依赖
    ```bash
    onnx 1.14.1
    onnx-graphsurgeon 0.5.2
    onnxsim 0.4.33
    ```