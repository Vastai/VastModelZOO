
## official

### step.1 获取预训练模型
```
link: https://github.com/MichaelFan01/STDC-Seg
branch: master
commit: 59ff37fbd693b99972c76fcefe97caa14aeb619f
```

克隆官方仓库后，将[export.py](./export.py)文件移动至原仓库工程目录下，配置相应参数，执行即可获得onnx和torchscript。


### step.2 准备数据集
- 下载[cityscapes](https://www.cityscapes-dataset.com/)数据集，解压

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_stdc.yaml](../build_in/build/official_stdc.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd stdc
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_stdc.yaml
    ```
   
### step.4 模型推理

    - 参考：[vsx_inference.py](../build_in/vsx/python/vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/vsx_inference.py \
        --image_dir  /path/to/cityscapes/leftImg8bit/val \
        --model_prefix_path deploy_weights/official_stdc_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-stdc1-vdsp_params.json \
        --mask_dir /path/to/cityscapes/gtFine/val \
        --color_txt ../source_code/cityscapes_colors.txt \
        --save_dir ./infer_output \
        --device 0
    ```

    ```
    # int8
    validation pixAcc: 89.028, mIoU: 50.751

    # fp16
    validation pixAcc: 91.767, mIoU: 55.671
    ```


### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-stdc1-vdsp_params.json](../build_in/vdsp_params/official-stdc1-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_stdc_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-stdc1-vdsp_params.json \
        -i 1 p 1 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../build_in/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../build_in/vdsp_params/image2npz.py \
        --dataset_path /path/to/cityscapes/leftImg8bit/val \
        --target_path  /path/to/cityscapes/leftImg8bit/val_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果
    ```bash
    vamp -m deploy_weights/official_stdc_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-stdc1-vdsp_params.json \
        -i 1 p 1 -b 1 \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)
    ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
        --src_dir /path/to/cityscapes/leftImg8bit/val \
        --gt_dir /path/to/cityscapes/gtFine/val \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir ./npz_output \
        --input_shape 512 512 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```
