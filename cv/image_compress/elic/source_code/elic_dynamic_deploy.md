### step.1 获取预训练模型

#### 获取官方pth模型
- 下载模型[ELIC_0450_ft_3980_Plateau.pth.tar](https://drive.google.com/file/d/1uuKQJiozcBfgGMJ8CfM6lrXOZWv6RUDN/view?usp=sharing)

#### onnx模型导出
- 下载官方工程代码
```
link: https://github.com/VincentChandelier/ELiC-ReImplemetation
branch: master
commit: 92a9ece1e6e1a188a12dfd7a58f9b51c554f9f2d
```

- 将[Network.py](./Network.py)文件放入工程目录，覆盖掉原来的同名文件
- 将[merge_ga_ha.py](./merge_ga_ha.py)文件放入工程目录
- 将[onnx_sub_1280_2048_ce.py](./onnx_sub_1280_2048_ce.py)文件放入工程目录
- 先分别导出ga/ha模型，执行命令：
```
# 数据集下载[kodak](https://drive.google.com/file/d/1Fst3a0naKWx28zX--kDB5G_T6Kyec9R6/view?usp=sharing)
python Inference.py --dataset ./kodak/ --output_path ELIC_0450_ft_3980_Plateau -p ./ELIC_0450_ft_3980_Plateau.pth.tar --patch 64
```
- 合并ga/ga模型，执行命令
```
python merge_ga_ha.py
```

- 动态模型需要将gs模型拆分为gs0和gs，执行命令
```
python onnx_sub_1280_2048_ce.py
```

- 注意python环境如下：
```
compressai==1.1.5
thop==0.1.1.post2209072238
ptflops==0.7.3
timm==1.0.3
torch==2.3.0
torchvision==0.18.0
onnx==1.16.0
onnxruntime==1.18.0
onnxsim==0.4.36
```


### step.2 准备数据集
- [测试数据集kodak](https://drive.google.com/file/d/1Fst3a0naKWx28zX--kDB5G_T6Kyec9R6/view?usp=sharing)

### step.3 模型转换
- 需要注意当前只支持FP16的模型。

1. 根据具体模型，修改编译配置
    - 以下三个模型仅支持fp16推理
    - [elic_dynamic_gaha.yaml](../build_in/build/elic_dynamic_gaha.yaml)
    - [elic_dynamic_gs.yaml](../build_in/build/elic_dynamic_gs.yaml)
    - [elic_dynamic_gs0.yaml](../build_in/build/elic_dynamic_gs0.yaml)
    - [elic_dynamic_hs_chunk.yaml](../build_in/build/elic_dynamic_hs_chunk.yaml)
        
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 执行转换
    ```bash
    cd elic
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/elic_dynamic_gaha.yaml
    vamc compile ../build_in/build/elic_dynamic_gs.yaml
    vamc compile ../build_in/build/elic_dynamic_gs0.yaml
    vamc compile ../build_in/build/elic_dynamic_hs_chunk.yaml
    ```

### step.4 模型推理
1. runstream
    - 获取[tensorize_ext_op](../../common/elf/tensorize_ext_op)
    ```bash
    python3 ../build_in/vsx/python/dynamic_elic_inference.py  \
        --gaha_model_info deploy_weights/elic_dynamic_gaha_run_stream_fp16/elic_dynamic_gaha_run_stream_fp16_module_info.json \
        --gaha_vdsp_params   ../build_in/vdsp_params/elic_compress_gaha_rgbplanar.json \
        --hs_model_info deploy_weights/elic_dynamic_hs_chunk_run_stream_fp16/elic_dynamic_hs_chunk_run_stream_fp16_module_info.json \
        --gs0_model_info deploy_weights/elic_dynamic_gs0_run_stream_fp16/elic_dynamic_gs0_run_stream_fp16_module_info.json \
        --gs_model_info deploy_weights/elic_dynamic_gs_run_stream_fp16/elic_dynamic_gs0_run_stream_fp16_module_info.json \
        --torch_model  /path/to/ELIC_0450_ft_3980_Plateau.pth.tar \
        --tensorize_elf_path ../../common/elf/tensorize_ext_op \
        --device_id  0 \
        --dataset_path /path/to/data/datasets/Kodak/ \
        --dataset_output_path dataset_outputs
    ```

    ```
    Ave Compress time:596.7426829867892 ms
    Ave Decompress time:648.1502321031359 ms
    Ave PNSR:35.357761837544075
    ```
