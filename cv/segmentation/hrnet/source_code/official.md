
## official

### step.1 获取预训练模型
```
link: https://github.com/HRNet/HRNet-Semantic-Segmentation
branch: pytorch-v1.1
commit: 88419ab18813f2c9193985e2d4d31d3d07abe839
```

克隆官方仓库后，将[export.py](./official/export.py)文件移动至`{git_dir}/tools`目录下，配置相应参数，执行即可获得onnx和torchscript。


### step.2 准备数据集
- 下载[cityscapes](https://www.cityscapes-dataset.com/)数据集，解压

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_hrnet.yaml](../build_in/build/official_hrnet.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd hrnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_hrnet.yaml
    ```
   
### step.4 模型推理
1. runstream
    - 参考：[vsx_inference.py](../build_in/vsx/python/vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/vsx_inference.py \
        --image_dir  /path/to/cityscapes/leftImg8bit/val \
        --model_prefix_path deploy_weights/official_hrnet_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-hrnet_w48-vdsp_params.json \
        --mask_dir /path/to/cityscapes/gtFine/val \
        --save_dir ./runstream_output \
        --color_txt ../source_code/official/cityscapes_colors.txt \
        --device 0
    ```

    ```
    # int8
    validation pixAcc: 93.720, mIoU: 64.230

    # fp16
    validation pixAcc: 93.786, mIoU: 65.358
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-hrnet_w48-vdsp_params.json](../build_in/vdsp_params/official-hrnet_w48-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/official_hrnet_run_stream_fp16/mod \
    --vdsp_params ../build_in/vdsp_params/official-hrnet_w48-vdsp_params.json \
    -i 1 p 1 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path /path/to/cityscapes/leftImg8bit/val \
    --target_path  /path/to/cityscapes/leftImg8bit/val_npz \
    --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_hrnet_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/official-hrnet_w48-vdsp_params.json \
        -i 1 p 1 -b 1 \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)，：
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
