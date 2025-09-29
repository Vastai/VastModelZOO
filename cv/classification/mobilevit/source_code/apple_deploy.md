## Deploy
### step.1 获取模型
- 原始仓库
    ```bash
    git clone git@github.com:apple/ml-cvnets.git
    ```
- 原始权重：[mobilevit_s.pt](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt)

- onnx导出
    - pytorch导出onnx参考：[export_onnx.py](./export_onnx.py)
    > - 环境依赖参考Tips
    - onnx：[mobilevit-small](https://drive.google.com/drive/folders/10tZUDbEXoBvIAuEmvU3WDXuxLq50o23M?usp=sharing)
    - 执行[convert_mobilevit_onnx.py](./convert_mobilevit_onnx.py)，将onnx转为固定batch、onnxsim以及替换custom odsp算子
        ```bash
        python convert_mobilevit_onnx.py
        ```

### step.2 获取数据集
- [校准数据集](https:/image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https:/image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../common/label/imagenet.txt)
- [label_dict](../../common/label/imagenet1000_clsid_to_human.txt)

### step.3 模型转换
0. 环境准备
    ```bash
    # 需要配置odsp环境变量
    cd /path/to/odsp_plugin/vastai/
    sh build.sh
    export LD_LIBRARY_PATH=/path/to/odsp_plugin/vastai/odsp_plugin/protobuf/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/path/to/odsp_plugin/vastai/odsp_plugin/vastai/lib:$LD_LIBRARY_PATH
    ```

1. 根据具体模型修改配置文件
    - [mobilevit_apple.yaml](../build_in/build/mobilevit_apple.yaml)：

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 模型编译

    ```bash
    cd mobilevit
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/mobilevit_apple.yaml
    ```

### step.4 模型推理
1. runstream
    - 执行如下命令进行runstream推理, 参考[infer_mobilevit.py](../build_in/vsx/python/infer_mobilevit.py)
    ```
    python ../build_in/vsx/python/infer_mobilevit.py \
        -m deploy_weights/apple_mobilevit_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/apple-mobilevit_b-vdsp_params.json \
        --device_id 0 \
        --label_file imagenet.txt \
        --dataset_filelist ILSVRC2012_img_val_filelist.txt \
        --dataset_root /path/to/cls/ImageNet \
        --dataset_output_file runstream_result.txt
    ```

    - 基于上一步vsx推理的结果，进行精度评估，参考[eval_topk.py](../../common/eval/eval_topk.py)
    ```
    python ../../common/eval/eval_topk.py runstream_result.txt
    ```

    ```
    # fp16
    top1_rate: 63.568 top5_rate: 85.606
    ```

### step.5 性能测试
1. 使用[infer_mobilevit_prof.py](../build_in/vsx/python/infer_mobilevit_prof.py)进行benchmark
    ```bash
    python3 ../build_in/vsx/python/infer_mobilevit_prof.py \
        -m deploy_weights/apple_mobilevit_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/apple-mobilevit_b-vdsp_params.json \
        --device_ids [0] \
        --batch_size 1 \
        --instance 1 \
        --iterations 300 \
        --shape [3,224,224] \
        --percentiles [50,90,95,99] \
        --input_host 1 \
        --queue_size 1

    ```

## Tips
- 当前mobilevit模型仅支持fp16推理，不支持int8推理
- python requirements如下
    ```bash
    # recommend python3.8
    onnx 1.16.0
    onnxruntime 1.18.0
    onnxsim 0.4.36
    onnx-graphsurgeon 0.5.2
    ```
