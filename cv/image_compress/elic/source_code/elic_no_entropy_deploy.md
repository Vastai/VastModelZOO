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

- 将[Export_no_entropy_onnx.py](./Export_no_entropy_onnx.py)文件放入工程目录
```
# 512x512 no_entropy模型导出
python Export_no_entropy_onnx.py \
    --pth_model ./ELIC_0450_ft_3980_Plateau.pth.tar \
    --model_input_shape 1,3,512,512 \
    --onnx_out_path ./elic_no_entropy_512x512.onnx

# 1280x2048 no_entropy模型导出
python Export_no_entropy_onnx.py \
    --pth_model ./ELIC_0450_ft_3980_Plateau.pth.tar \
    --model_input_shape 1,3,1280,2048 \
    --onnx_out_path ./elic_no_entropy_1280x2048.onnx
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
    - [elic_no_entropy_512_512.yaml](../build_in/build/elic_no_entropy_512_512.yaml)
    - [elic_no_entropy_1280_2048.yaml](../build_in/build/elic_no_entropy_1280_2048.yaml)
        
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 执行转换
    ```bash
    cd elic
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/elic_no_entropy_512_512.yaml
    vamc compile ../build_in/build/elic_no_entropy_1280_2048.yaml
    ```

### step.4 模型推理

- 测试512x512模型
    ```bash
    python3 ../build_in/vsx/python/elic_no_entropy_inference.py  \
        --model_prefix deploy_weights/elic_no_entropy_512_512_fp16/mod \
        --vdsp_params  ../build_in/vdsp_params/elic_compress_gaha_rgbplanar.json \
        --device_id  0 \
        --dataset_path /path/to/datasets/Kodak-512/ \
        --dataset_output_path dataset_outputs_512x512
    ```

    ```
    Ave Compress time:107.34399159749348 ms
    Ave PNSR:37.706234893066075
    Ave bbp:0.843717660754919
    ```

- 测试1280x2048模型
    ```bash
    python3 ../build_in/vsx/python/elic_no_entropy_inference.py  \
        --model_prefix deploy_weights/elic/elic_no_entropy_1280_2048_fp16/mod \
        --vdsp_params  ../build_in/vdsp_params/elic_compress_gaha_rgbplanar.json \
        --device_id  0 \
        --dataset_path /path/to/datasets/Kodak_1280_2048/ \
        --dataset_output_path dataset_outputs_1280x2048
    ```

    ```
    Ave Compress time:1303.3234576384227 ms
    Ave PNSR:43.009992020884276
    Ave bbp:0.262036203717192  
    ```

### 性能测试
1. 参考[elic_noentropy_prof.py](../build_in/vsx/python/elic_noentropy_prof.py)，测试image模型性能：
    - 测试512x512模型最大吞吐和最小时延
    ```
    python ../build_in/vsx/python/elic_noentropy_prof.py \
        --model_prefix deploy_weights/elic_no_entropy_512_512_fp16/mod \
        --vdsp_params  ../build_in/vdsp_params/elic_no_entropy_gaha_512_512_rgbplanar.json \
        --instance 1 \
        --iterations 100 \
        --queue_size 1 \
        --batch_size 1
    ```

    - 测试1280x2048模型最大吞吐和最小时延
    ```
    python ../build_in/vsx/python/elic_noentropy_prof.py \
        --model_prefix deploy_weights/elic_no_entropy_1280_2048_fp16/mod \
        --vdsp_params  ../build_in/vdsp_params/elic_no_entropy_gaha_1024_2048_rgbplanar.json \
        --instance 1 \
        --iterations 10 \
        --batch_size 1  \
        --queue_size 1
    ```
