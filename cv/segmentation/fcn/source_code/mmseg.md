
## Build_In Deploy

### step.1 模型准备

1. 下载模型权重

    ```
    link: https://github.com/open-mmlab/mmsegmentation
    branch: v1.0.0rc2
    commit: 8a611e122d67b1d36c7929331b6ff53a8c98f539
    ```

2. 模型导出

    使用mmseg转换代码[pytorch2torchscript.py](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/pytorch2torchscript.py)，命令如下
    ```bash
    python tools/pytorch2torchscript.py  \
        configs/fcn/fcn_r50-d8_512x512_20k_voc12aug.py \
        --checkpoint ./pretrained/mmseg/ann/fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth \
        --output-file ./onnx/mmseg/ann/torchscript/fcn_r50_d8_20k-512.torchscript.pt \
        --shape 512 512
    ```
    > onnx在build时会报错

### step.2 准备数据集
- 下载[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)数据集，解压

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [mmseg_fcn.yaml](../build_in/build/mmseg_fcn.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    vamc compile ../build_in/build/mmseg_fcn.yaml
    ```

### step.4 模型推理
1. runstream推理：[mmseg_vsx_inference.py](../build_in/vsx/mmseg_vsx_inference.py)
    ```bash
    python ../build_in/vsx/mmseg_vsx_inference.py \
        --image_dir  /path/to/VOC2012/JPEGImages_val \
        --model_prefix_path deploy_weights/mmseg_fcn_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/mmseg-fcn_r50_d8_20k-vdsp_params.json \
        --mask_dir /path/to/SegmentationClass \
        --save_dir ./runstream_output \
        --color_txt ../source_code/awesome/voc2012_colors.txt \
        --device 0
    ```

    ```
    # int8
    validation pixAcc: 92.628, mIoU: 67.355

    # fp16
    validation pixAcc: 92.534, mIoU: 67.019
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[mmseg-fcn_r50_d8_20k-vdsp_params.json](../build_in/vdsp_params/mmseg-fcn_r50_d8_20k-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/mmseg_fcn_run_stream_fp16/mod \
    --vdsp_params ../build_in/vdsp_params/mmseg-fcn_r50_d8_20k-vdsp_params.json \
    -i 2 p 2 -b 1 -s [3,512,512]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../build_in/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`，注意只转换`VOC2012/ImageSets/Segmentation/val.txt`对应的验证集图像（配置相应路径）：
    ```bash
    python ../build_in/vdsp_params/image2npz.py \
        --dataset_path VOC2012/JPEGImages \
        --target_path  VOC2012/JPEGImages_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/mmseg_fcn_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/mmseg-fcn_r50_d8_20k-vdsp_params.json \
        -i 2 p 2 -b 1 -s [3,512,512] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```
    
    - 解析npz结果并统计精度, 参考:[mmseg_vamp_eval.py](../build_in/vdsp_params/mmseg_vamp_eval.py)
    ```bash
    python ../build_in/vdsp_params/mmseg_vamp_eval.py \
        --src_dir VOC2012/JPEGImages_val \
        --gt_dir VOC2012/SegmentationClass \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir ./npz_output \
        --input_shape 512 512 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```
