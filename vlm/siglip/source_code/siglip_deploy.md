## Deploy
### step.1 工程源码

```
https://github.com/google-research/big_vision
6d6c28a9634fd2f48f0f505f112d063dfc9bdf96
```

### step.2 onnx导出
- 参考[export_onnx.py](./export_onnx.py)导出onnx模型
- 注意该脚本会自动下载hf的原始模型
```
python export_onnx.py
```

### step.3 获取数据集
> 以ISLVRC2012来检验siglip在图像分类领域的能力
- [校准数据集](https:/image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https:/image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../../cv/classification/common/label/imagenet.txt)
- [label_dict](../../../cv/classification/common/label/imagenet1000_clsid_to_human.txt)

### step.4 模型转换

1. 根据具体模型，修改编译配置
    - 当前模型只支持fp16
    - [siglip_image_encoder.yaml](../build_in/build/siglip_image_encoder.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 模型编译
    ```bash
    cd siglip
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/siglip_image_encoder.yaml
    ```

### step.5 模型推理
1. runstream
    - [elf文件](../../../cv/classification/common/elf/)
    - 参考：[siglip_sample.py](../build_in/vsx/python/siglip_sample.py)
    ```bash
    python3 ../build_in/vsx/python/siglip_sample.py \
        --model_prefix deploy_weights/siglip_image_run_stream_fp16/mod \
        --onnx_path /path/to/siglip-instruct-sim.onnx \
        --norm_elf path/to/elf/normalize \
        --space2depth_elf path/to/elf/space_to_depth \
        --device_id 0 \
        --dataset_root /path/to/ILSVRC2012_img_val
    ```

    - 注意当前精度以vacc模型输出特征值与onnx模型输出特征值的cosin距离为准
    ```
    Average Cosine Similarity: 0.9999999865679674
    Maximum Cosine Similarity: 1
    Minimum Cosine Similarity: 0.9999998807907104
    ```

### step.6 性能测试
1. 参考[siglip_image_prof.py](../build_in/vsx/python/siglip_image_prof.py)测试clip_image的性能
    ```bash
    python3 ../build_in/vsx/python/siglip_image_prof.py \
        --model_prefix deploy_weights/siglip_image_run_stream_fp16/mod \
        --norm_elf /path/to/elf/normalize \
        --space2depth_elf /path/to/elf/space_to_depth \
        --device_ids  [0] \
        --batch_size  1 \
        --instance 1 \
        --iterations 100 \
        --percentiles "[50,90,95,99]" \
        --input_host 1 \
        --queue_size 1
    ```

### appending
- 当前仅支持图像编码器部分