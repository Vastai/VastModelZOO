### step.1 获取模型
```bash
pip install PaddlePaddle==2.3.2  Paddle2ONNX==1.0.0
paddle2onnx  --model_dir /path/to/cspnet_paddle_model/ \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file model.onnx \
            --enable_dev_version False \
            --opset_version 10
```

### step.2 获取数据集
- [校准数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../common/label/imagenet.txt)
- [label_dict](../../common/label/imagenet1000_clsid_to_human.txt)

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [ppcls_cspnet.yaml](../build_in/build/ppcls_cspnet.yaml)

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd cspnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ppcls_cspnet.yaml
    ```

### step.4 模型推理
- 参考：[classification.py](../../common/vsx/classification.py)
    ```bash
    python ../../common/vsx/classification.py \
        --infer_mode sync \
        --file_path path/to/ILSVRC2012_img_val \
        --model_prefix_path deploy_weights/ppcls_cspnet_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/ppcls-cspdarknet53-vdsp_params.json \
        --label_txt path/to/imagenet.txt \
        --save_dir ./infer_output \
        --save_result_txt result.txt \
        --device 0
    ```

- 精度评估
    ```
    python ../../common/eval/eval_topk.py ./infer_output/result.txt
    ```

    ```
    # fp16
    top1_rate: 77.652 top5_rate: 93.94

    # int8
    top1_rate: 58.024 top5_rate: 81.28
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置[ppcls-cspdarknet53-vdsp_params.json](../build_in/vdsp_params/ppcls-cspdarknet53-vdsp_params.json)
    ```
    vamp -m deploy_weights/ppcls_cspnet_int8/mod --vdsp_params ../build_in/vdsp_params/ppcls-cspdarknet53-vdsp_params.json  -i 8 -p 1 -b 2
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，生成推理数据`npz`以及对应的`dataset.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path ILSVRC2012_img_val --target_path  input_npz  --text_path imagenet_npz.txt
    ```

    - vamp推理获取npz文件
    ```bash
    ./vamp -m deploy_weights/ppcls_cspnet_int8/mod --vdsp_params ../build_in/vdsp_params/ppcls-cspdarknet53-vdsp_params.json  -i 8 -p 1 -b 22 -s [3,256,256] --datalist imagenet_npz.txt --path_output output
    ```

    - 解析npz文件用于精度评估，参考：[vamp_npz_decode.py](../../common/eval/vamp_npz_decode.py)
    ```bash
    python  ../../common/eval/vamp_npz_decode.py imagenet_npz.txt output imagenet_result.txt imagenet.txt
    ```

    - 精度评估，参考：[eval_topk.py](../../common/eval/eval_topk.py)
    ```bash
    python ../../common/eval/eval_topk.py imagenet_result.txt
    ```
