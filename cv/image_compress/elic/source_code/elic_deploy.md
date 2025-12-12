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
- 先分别导出ga/ha模型，执行命令：
```
# 数据集下载[kodak](https://drive.google.com/file/d/1Fst3a0naKWx28zX--kDB5G_T6Kyec9R6/view?usp=sharing)
python Inference.py --dataset ./kodak/ --output_path ELIC_0450_ft_3980_Plateau -p ./ELIC_0450_ft_3980_Plateau.pth.tar --patch 64
```
- 合并ga/ga模型，执行命令
```
python merge_ga_ha.py
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
    - [elic_gaha.yaml](../build_in/build/elic_gaha.yaml)
    - [elic_gs.yaml](../build_in/build/elic_gs.yaml)
    - [elic_hs_chunk.yaml](../build_in/build/elic_hs_chunk.yaml)
        
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 执行转换
    ```bash
    cd elic_pp
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/elic_gaha.yaml
    vamc compile ../build_in/build/elic_gs.yaml
    vamc compile ../build_in/build/elic_hs_chunk.yaml
    ```

### step.4 模型推理

- 获取[tensorize_ext_op](../../common/elf/tensorize_ext_op)，执行[elic_inference.py](../build_in/vsx/python/elic_inference.py)脚本，执行命令如下：
    ```bash
    python3 ../build_in/vsx/python/elic_inference.py  \
        --gaha_model_prefix  deploy_weighst/elic_gaha_fp16/mod \
        --gaha_vdsp_params  ../build_in/vdsp_params/elic_compress_gaha_rgbplanar.json \
        --hs_model_prefix  deploy_weighst/elic_hs_chunk_fp16/mod \
        --gs_model_prefix   deploy_weighst/elic_gs_fp16/mod \
        --torch_model  /path/to/ELIC_0450_ft_3980_Plateau.pth.tar \
        --tensorize_elf_path ../../common/elf/tensorize_ext_op \
        --device_id  0 \
        --dataset_path /path/to/datasets/Kodak-512/ \
        --dataset_output_path infer_outputs
    ```

    ```
    Ave Compress time:596.7426829867892 ms
    Ave Decompress time:648.1502321031359 ms
    Ave PNSR:35.357761837544075
    ```

### Tips
- 由于此网络既包含vacc推理过程也包含pytorch过程，所以暂不提供vacc的性能测试脚本。