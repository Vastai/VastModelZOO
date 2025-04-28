### step.1 获取模型
```bash
pip install PaddlePaddle==2.3.2  Paddle2ONNX==1.0.0

paddle2onnx  --model_dir /path/to/paddle_model/ \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file model.onnx \
            --enable_dev_version False \
            --opset_version 10
```

### step.2 获取数据集
- [校准数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../common/label//imagenet.txt)
- [label_dict](../../common/label//imagenet1000_clsid_to_human.txt)

### step.3 模型转换

1. 参考瀚博训推软件生态链文档，获取模型转换工具: [vamc v3.0+](../../../../docs/vastai_software.md)

2. 根据具体模型，修改编译配置
    - [ppcls_mobilenetv3.yaml](../build_in/build/ppcls_mobilenetv3.yaml)
    
    > - runmodel推理，编译参数`backend.type: tvm_runmodel`
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

3. 模型编译

    ```bash
    cd mobilenetv3
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ppcls_mobilenetv3.yaml
    ```

### step.4 模型推理

1. 参考瀚博训推软件生态链文档，获取模型推理工具：[vaststreamx v2.8+](../../../../docs/vastai_software.md)

2. runstream
    - 参考：[classification.py](../../common/vsx/classification.py)
    ```bash
    python ../../common/vsx/classification.py \
        --infer_mode sync \
        --file_path path/to/ILSVRC2012_img_val \
        --model_prefix_path deploy_weights/ppcls_mobilenetv3_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/ppcls-mobilenet_v3_small_x1.0-vdsp_params.json \
        --label_txt path/to/imagenet.txt \
        --save_dir ./runstream_output \
        --save_result_txt result.txt \
        --device 0
    ```

    - 精度评估
    ```
    python ../../common/eval/eval_topk.py ./runstream_output/result.txt
    ```

    ```
    # mobilenetv3_small_x1.0 模型精度

    # fp16
    top1_rate: 67.2 top5_rate: 87.24

    # int8
    top1_rate: 0.126 top5_rate: 0.64
    ```

### step.5 性能测试
1. 参考瀚博训推软件生态链文档，获取模型性能测试工具：[vamp v2.4+](../../../../docs/vastai_software.md)

2. 性能测试
    - 配置[ppcls-mobilenet_v3_small_x1.0-vdsp_params.json](../build_in/vdsp_params/ppcls-mobilenet_v3_small_x1.0-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/ppcls_mobilenetv3_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/ppcls-mobilenet_v3_small_x1.0-vdsp_params.json  -i 8 -p 1 -b 2 -s [3,224,224]
    ```

3. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致
    
    - 数据准备，生成推理数据`npz`以及对应的`dataset.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path ILSVRC2012_img_val --target_path  input_npz  --text_path imagenet_npz.txt
    ```

    - vamp推理获取npz文件
    ```
    vamp -m deploy_weights/ppcls_mobilenetv3_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/ppcls-mobilenet_v3_small_x1.0-vdsp_params.json  -i 8 -p 1 -b 22 -s [3,224,224] --datalist imagenet_npz.txt --path_output output
    ```

    - 解析输出结果用于精度评估，参考：[vamp_npz_decode.py](../../common/eval/vamp_npz_decode.py)
    ```bash
    python  ../../common/eval/vamp_npz_decode.py imagenet_npz.txt output imagenet_result.txt imagenet.txt
    ```
    
    - 精度评估，参考：[eval_topk.py](../../common/eval/eval_topk.py)
    ```bash
    python ../../common/eval/eval_topk.py imagenet_result.txt
    ```

### appending
- ppcls：FP16精度正常，INT8掉点严重
    ```bash
    MobileNetV3_large_x0_5
    benchmark:{"top1": 69.24, "top5": 88.52}
    fp16:{"top1": 67.856, "top5": 87.774}
    int8:{"top1": 0.62, "top5": 2.126}

    MobileNetV3_small_x1_25
    benchmark:{"top1": 70.67, "top5": 89.51}
    fp16:{"top1": 69.6560, "top5": 88.9060}
    int8:{"top1": 28.406, "top5": 52.024}

    MobileNetV3_small_x1_0
    benchmark:{"top1": 68.24, "top5": 88.06}
    fp16：{"top1": 66.96, "top5": 87.11}
    int8:{"top1": 0.128, "top5": 0.7940}
    ```
