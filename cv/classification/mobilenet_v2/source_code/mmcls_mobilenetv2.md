### step.1 获取模型
mmcls框架参考 [mmclassification](https://github.com/open-mmlab/mmclassification),可使用如下位置的pytorch2onnx.py或pytorch2torchscript.py转成相应的模型
```bash
cd mmclassification

python tools/deployment/pytorch2onnx.py \
    --config configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py \
    --checkpoint weights/mobilenet_v2.pth \
    --output-file output/mobilenet_v2.onnx \
```

### step.2 获取数据集
- [校准数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../common/label//imagenet.txt)
- [label_dict](../../common/label//imagenet1000_clsid_to_human.txt)

### step.3 模型转换

1. 参考瀚博训推软件生态链文档，获取模型转换工具: [vamc v3.0+](../../../../docs/vastai_software.md)

2. 根据具体模型，修改编译配置
    - [mmcls_mobilenetv2.yaml](../build_in/build/mmcls_mobilenetv2.yaml)
    
    > - runmodel推理，编译参数`backend.type: tvm_runmodel`
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

3. 模型编译

    ```bash
    cd mobilenet_v2
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/mmcls_mobilenetv2.yaml
    ```

### step.4 模型推理

1. 参考瀚博训推软件生态链文档，获取模型推理工具：[vaststreamx v2.8+](../../../../docs/vastai_software.md)

2. runstream
    - 参考：[classification.py](../../common/vsx/classification.py)
    ```bash
    python ../../common/vsx/classification.py \
        --infer_mode sync \
        --file_path path/to/ILSVRC2012_img_val \
        --model_prefix_path deploy_weights/mmcls_mobilenetv2_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/mmcls-mobilenet_v2-vdsp_params.json \
        --label_txt path/to/imagenet.txt \
        --save_dir ./runstream_output \
        --save_result_txt result.txt \
        --device 0
    ```

    - 精度评估
    ```
    python ../../common/eval/eval_topk.py ./runmstream_output/mod.txt
    ```

    ```
    # fp16
    top1_rate: 70.244 top5_rate: 89.4

    # int8
    top1_rate: 68.764 top5_rate: 88.572
    ```

### step.5 性能测试
1. 参考瀚博训推软件生态链文档，获取模型性能测试工具：[vamp v2.4+](../../../../docs/vastai_software.md)

2. 性能测试
    - 配置[mmcls-mobilenet_v2-vdsp_params.json](../build_in/vdsp_params/mmcls-mobilenet_v2-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/mmcls_mobilenetv2_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/mmcls-mobilenet_v2-vdsp_params.json  -i 8 -p 1 -b 2 -s [3,224,224]
    ```

3. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致
    
    - 数据准备，生成推理数据`npz`以及对应的`dataset.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path ILSVRC2012_img_val --target_path  input_npz  --text_path imagenet_npz.txt
    ```

    - vamp推理获取npz文件
    ```
    vamp -m deploy_weights/mmcls_mobilenetv2_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/mmcls-mobilenet_v2-vdsp_params.json  -i 8 -p 1 -b 22 -s [3,224,224] --datalist imagenet_npz.txt --path_output output
    ```

    - 解析输出结果用于精度评估，参考：[vamp_npz_decode.py](../../common/eval/vamp_npz_decode.py)
    ```bash
    python  ../../common/eval/vamp_npz_decode.py imagenet_npz.txt output imagenet_result.txt imagenet.txt
    ```
    
    - 精度评估，参考：[eval_topk.py](../../common/eval/eval_topk.py)
    ```bash
    python ../../common/eval/eval_topk.py imagenet_result.txt
    ```