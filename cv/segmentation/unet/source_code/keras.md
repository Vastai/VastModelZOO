## Build_In Deploy

### step.1 模型准备
1. 下载模型权重

    ```
    link：https://github.com/zhixuhao/unet
    branch: master
    commit: b45af4d458437d8281cc218a07fd4380818ece4a
    ```
    
2. 模型导出

- 拉取原始仓库，将[keras](./source_code/keras)文件夹下脚本，替换原始仓库对应文件
- 原始仓库为单通道模型，我们重新训练3通道模型，基于[main.py](./keras/main.py)，保存`h5`含有网络结构和权重


### step.2 准备数据集
- 下载[isbi](https://github.com/zhixuhao/unet/tree/master/data/membrane)数据集

### step.3 模型转换
1. 根据具体模型修改模型转换配置文件
    - [keras_unet.yaml](../build_in/build/keras_unet.yaml)

    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd unet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/keras_unet.yaml
    ```
    
### step.4 模型推理
1. 参考：[keras_vsx.py](../build_in/vsx/python/keras_vsx.py)，修改参数并运行如下脚本
    ```bash
    python ../build_in/vsx/python/keras_vsx.py \
        --file_path  /path/to/isbi/train/image \
        --model_prefix_path deploy_weights/keras_unet_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/keras-unet-vdsp_params.json \
        --gt_path /path/to/isbi/train/label \
        --save_dir ./infer_output \
        --device 0
    ```

### step.5 性能精度测试
1. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path isbi/train/image \
    --target_path  isbi/train/image_npz \
    --text_path npz_datalist.txt
    ```

2. 性能测试，配置vdsp参数[keras-unet-vdsp_params.json](../build_in/vdsp_params/keras-unet-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/keras_unet_int8/mod \
    --vdsp_params ../build_in/vdsp_params/keras-unet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256]
    ```

> 可选步骤，和step.4的精度测试基本一致

3. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/keras_unet_int8/mod \
    --vdsp_params ../build_in/vdsp_params/keras-unet-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,256,256] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```

4. [keras-vamp_eval.py](../build_in/vdsp_params/keras-vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../build_in/vdsp_params/keras-vamp_eval.py \
    --src_dir isbi/train/image \
    --gt_dir isbi/train/label \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir output/unet \
    --input_shape 256 256 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


## Tips
- 编译keras类型的模型，需安装对应依赖
    ```bash
    pip install keras==2.8.0 tensorflow==2.8.0
    ```