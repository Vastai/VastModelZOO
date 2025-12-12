### step.1 准备预训练模型
1. official
    ```bash
    git clone https://github.com/apple/ml-mobileone.git
    cp source_code/export_onnx.py ml-mobileone
    cd ml-mobileone
    python export_onnx.py
    ```

### step.2 准备数据集
- [校准数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../common/label/imagenet.txt)
- [label_dict](../../common/label/imagenet1000_clsid_to_human.txt)

### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [official_mobileone.yaml](../build_in/build/official_mobileone.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd mobileone
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_mobileone.yaml
    ```

### step.4 模型推理

- 参考：[classification.py](../../common/vsx/python/classification.py)
    ```bash
    python ../../common/vsx/python/classification.py \
        --file_path path/to/ILSVRC2012_img_val \
        --model_prefix_path deploy_weights/official_mobileone_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-mobileone_s4-vdsp_params.json \
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
    top1_rate: 78.572 top5_rate: 93.788

    # int8
    top1_rate: 73.252 top5_rate: 91.266
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置[official-mobileone_s4-vdsp_params.json](../build_in/vdsp_params/official-mobileone_s4-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_mobileone_int8/mod --vdsp_params ../build_in/vdsp_params/official-mobileone_s4-vdsp_params.json  -i 8 -p 1 -b 2 -s [3,224,224]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；
    
    - 数据准备，生成推理数据`npz`以及对应的`dataset.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path ILSVRC2012_img_val --target_path  input_npz  --text_path imagenet_npz.txt
    ```

    - vamp推理获取npz文件
    ```
    vamp -m deploy_weights/official_mobileone_int8/mod --vdsp_params ../build_in/vdsp_params/official-mobileone_s4-vdsp_params.json  -i 8 -p 1 -b 22 -s [3,224,224] --datalist imagenet_npz.txt --path_output output
    ```

    - 解析输出结果用于精度评估，参考：[vamp_npz_decode.py](../../common/eval/vamp_npz_decode.py)
    ```bash
    python  ../../common/eval/vamp_npz_decode.py imagenet_npz.txt output imagenet_result.txt imagenet.txt
    ```
    
    - 精度评估，参考：[eval_topk.py](../../common/eval/eval_topk.py)
    ```bash
    python ../../common/eval/eval_topk.py imagenet_result.txt
    ```
