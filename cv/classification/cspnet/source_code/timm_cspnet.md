### step.1 获取模型
```bash
pip install timm==0.6.5
python ../common/utils/export_timm_torchvision_model.py --model_library timm  --model_name cspresnet50 --save_dir ./onnx  --size 256 --pretrained_weights xxx.pth
```

### step.2 获取数据集
- [校准数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../common/label/imagenet.txt)
- [label_dict](../../common/label/imagenet1000_clsid_to_human.txt)

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [timm_cspnet.yaml](../build_in/build/timm_cspnet.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd cspnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/timm_cspnet.yaml
    ```
   

### step.4 模型推理
1. runstream
    - 参考：[classification.py](../../common/vsx/classification.py)
    ```bash
    python ../../common/vsx/classification.py \
        --file_path path/to/ILSVRC2012_img_val \
        --model_prefix_path deploy_weights/timm_cspnet_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/timm-cspdarknet53-vdsp_params.json \
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
    # int8
    top1_rate: 78.842 top5_rate: 94.325
    
    # fp16
    top1_rate: 79.19 top5_rate: 94.496
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置[timm-cspresnet50-vdsp_params.json](../build_in/vdsp_params/timm-cspresnet50-vdsp_params.json)
    ```bash
    vamp -m ./deploy_weights/timm_cspnet_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/timm-cspresnet50-vdsp_params.json  -i 8 -p 1 -b 2
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 准备数据，生成推理数据`npz`以及对应的`dataset.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path ILSVRC2012_img_val --target_path  input_npz  --text_path imagenet_npz.txt
    ```
    - vamp推理获取npz文件
    ```
    vamp -m ./deploy_weights/timm_cspnet_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/timm-cspresnet50-vdsp_params.json  -i 8 -p 1 -b 22 -s [3,256,256] --datalist imagenet_npz.txt --path_output output
    ```
    
    - 解析npz输出结果用于精度评估，参考：[vamp_npz_decode.py](../../common/eval/vamp_npz_decode.py)
    ```bash
    python  ../../common/eval/vamp_npz_decode.py imagenet_npz.txt output imagenet_result.txt imagenet.txt
    ```

    - 精度评估，参考：[eval_topk.py](../../common/eval/eval_topk.py)
    ```bash
    python ../../common/eval/eval_topk.py imagenet_result.txt
    ```