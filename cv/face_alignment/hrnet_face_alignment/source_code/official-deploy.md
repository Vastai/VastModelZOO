### step.1 获取预训练模型
1. 源repo未提供转换onnx的脚本， 可参考[pytorch2onnx.py](./pytorch2onnx.py)进行转换


    ```bash
    cd HRNet-Facial-Landmark-Detection
    python pytorch2onnx.py --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml \
        --model-file hrnetv2_pretrained/HR18-WFLW.pth
    ```

### step.2 准备数据集
- [校准数据集](https://wywu.github.io/projects/LAB/WFLW.html)
- [评估数据集](https://wywu.github.io/projects/LAB/WFLW.html)
    - 需要自己生成预处理后的数据，进入[工程](https://github.com/jhb86253817/PIPNet.git)，按如下步骤操作：
    ```bash
    cd lib
    #执行预处理脚本时需按照实际模型输入尺寸进行修改，本例中为256
    python preprocess.py WFLW
    ```

### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [official_hrnet.yaml](../build_in/build/official_hrnet.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd hrnet_face_alignment
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_hrnet.yaml
    ```

### step.4 模型推理

- 参考：[vsx_infer.py](../build_in/vsx/python/vsx_infer.py)
    ```bash
    python ../build_in/vsx/python/vsx_infer.py \
        --data_dir  /path/to/wflw/images \
        --npz_datalist  ./npz_datalist.txt \
        --model_prefix_path deploy_weights/official_hrnet_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/pytorch-hrnetv2-vdsp_params.json \
        --save_dir ./infer_output \
        --device 0
    ```

- 精度测试，参考：[npz_decode.py](../build_in/npz_decode.py)
    ```
    python ../build_in/npz_decode.py --result ./infer_output --gt /path/to/wflw_meta.json --debug true
    ```

    ```
    # fp16
    nme: 0.4712
    failure_rate_008: 1.0
    failure_rate_010: 1.0

    # int8
    nme: 0.4716
    failure_rate_008: 1.0
    failure_rate_010: 1.0
    ```

### step.5 性能精度测试
1. 性能测试
    配置[pytorch-hrnetv2-vdsp_params.json](../build_in/vdsp_params/pytorch-hrnetv2-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_hrnet_int8/mod --vdsp_params ../build_in/vdsp_params/pytorch-hrnetv2-vdsp_params.json -i 2 p 2 -b 2
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，输入数据是经过预处理后的npz， 使用如下代码生成npz_datalist.txt
    
    ```python
    import glob

    targ_path = "build_in/npz_data"
    file_list = glob.glob(targ_path+"/*.npz")
    with open("npz_datalist.txt", "w") as fw:
        for file in file_list:
            fw.write(file+'\n')
    ```

    - vamp推理获取npz结果
    ```bash
    vamp -m deploy_weights/official_hrnet_int8c/mod \
        --vdsp_params ../build_in/vdsp_params/pytorch-hrnetv2-vdsp_params.json \
        -i 1 p 1 -b 1 \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析vamp输出的npz文件，参考：[npz_decode.py](../build_in/npz_decode.py)，
    ```bash
    python ../build_in/npz_decode.py  \
        --result vamp_out \
        --gt  /path/to/wflw_meta.json \
        --npz-txt npz_datalist.txt
    ```
