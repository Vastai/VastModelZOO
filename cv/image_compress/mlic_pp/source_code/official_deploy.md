### step.1 获取预训练模型

#### 获取官方pth模型
- 下载模型[mlicpp_mse_q5_2960000.pth.tar](https://github.com/JiangWeibeta/MLIC?tab=readme-ov-file#pretrained-models)

##### onnx模型导出
1. 下载[官方工程代码](https://github.com/JiangWeibeta/MLIC.git)，commit号为：d001ea434d05b3e160b951ceafa6d9556b207326
2. 将source_code/mlicpp.py替换掉官方工程的MLIC++/models/mlicpp.py同名文件
3. 将merge_ga_ha.py文件复制到官方工程的MLIC++/playground/路径下
4. 准备好测试数据，只需要保留一张图片即可
5. 通过在单张图片推理的过程中，导出onnx模型
    - 导出onnx模型
    ```
    cd MLIC++/playground/ ;
    bash test.sh
    ```
    
    - 合并onnx模型
    ```
    cd MLIC++/playground/ ;
    python merge_ga_ha.py
    ```

    - 生成的compress_ga_ha_sim就是合并后的onnx模型，compress_hs_sim.onnx和decompress_gs_sim.onnx也是需要的

### step.2 准备数据集
- 由于模型的输入尺寸固定为[512,768]，原始kodak包含其他尺寸图片，所以从里面抽出尺寸为[512,768]的数据作为测试集
- [测试数据集kodak](https://drive.google.com/file/d/1Fst3a0naKWx28zX--kDB5G_T6Kyec9R6/view?usp=sharing)

### step.3 模型转换
- 需要注意当前只支持FP16的模型。

1. 根据具体模型，修改编译配置
    - 以下三个模型仅支持fp16推理
    - [compress_ga_ha_sim_512_768.yaml](../build_in/build/compress_ga_ha_sim_512_768.yaml)
    - [compress_hs_sim.yaml](../build_in/build/compress_hs_sim.yaml)
    - [decompress_gs_sim.yaml](../build_in/build/decompress_gs_sim.yaml)

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 执行转换
    ```bash
    cd mlic_pp
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/compress_ga_ha_sim_512_768.yaml
    vamc compile ../build_in/build/compress_hs_sim.yaml
    vamc compile ../build_in/build/decompress_gs_sim.yaml
    ```

### step.4 模型推理

- 参考: [mlic_inference.py](../build_in/vsx/python/mlic_inference.py)
    ```bash
    python3 ../build_in/vsx/python/mlic_inference.py  \
        --gaha_model_prefix  deploy_weights/compress_ga_ha_sim_512_768_fp16/mod \
        --gaha_vdsp_params  ../build_in/vdsp_params/mlic_compress_gaha_rgbplanar.json \
        --hs_model_prefix  deploy_weights/compress_hs_sim_fp16/mod \
        --gs_model_prefix   deploy_weights/decompress_gs_sim_fp16/mod \
        --torch_model  /path/to/mlicpp_mse_q5_2960000.pth.tar \
        --device_id  0 \
        --dataset_path /path/to/kodak \
        --dataset_output_path dataset_outputs
    ```

    ```
    Ave Compress time:596.7426829867892 ms
    Ave Decompress time:648.1502321031359 ms
    Ave PNSR:35.357761837544075
    ```

### Tips
- 由于算子支持原因，该算法只有gaha/hs/gs模型运行在vacc硬件上，其余模块使用官方的实现。
- 本工程的python依赖项如下：
```
CompressAI==1.2.0b3
pydantic==2.10.6
einops
timm
```