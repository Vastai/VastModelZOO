
## GPEN版本

### step.1 获取预训练模型
```
link: https://github.com/yangxy/GPEN/blob/main/face_parse/parse_model.py
branch: master
commit: c9cc29009b633788a77d782ba102cee913e3a349
```

基于[export_onnx.py](./gpen/face_parse/export_onnx.py)，转换模型。


### step.2 准备数据集
- 下载[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)数据集，按官方处理数据集


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_parsenet.yaml](../build_in/build/official_parsenet.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd parsenet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_parsenet.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考：[vsx_inference.py](../build_in/vsx/python/vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/vsx_inference.py \
        --image_dir  /path/to/CelebAMask-HQ/test_img \
        --model_prefix_path deploy_weights/official_parsenet_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/gpen-parsenet-vdsp_params.json \
        --mask_dir /path/to/CelebAMask-HQ/test_label \
        --save_dir ./runstream_output \
        --device 0
    ```

    <details><summary>点击查看精度</summary>

    ```
    # int8
    ----------------- Total Performance --------------------
    Overall Acc:     0.9181530063618344
    Mean Acc :       0.7476149007672467
    FreqW Acc :      0.8509179929308909
    Mean IoU :       0.6661899068917726
    Overall F1:      0.7664677255343008
    ----------------- Class IoU Performance ----------------
    background      : 0.8751358107656723
    skin    : 0.9107283591779487
    nose    : 0.8566283375418006
    eyeglass        : 0.6659175431207196
    left_eye        : 0.7573389295432673
    right_eye       : 0.756044914610085
    left_brow       : 0.7028084779747561
    right_brow      : 0.699635243931522
    left_ear        : 0.7286968359308529
    right_ear       : 0.6870223562027429
    mouth   : 0.7828061182251146
    upper_lip       : 0.7220229001227981
    lower_lip       : 0.7854557776744097
    hair    : 0.8584439647465846
    hat     : 0.21431260044765338
    earring : 0.2513102490085276
    necklace        : 0.005015337224803593
    neck    : 0.7865113666502132
    cloth   : 0.6117731080442076
    --------------------------------------------------------

    # fp16
    ----------------- Total Performance --------------------
    Overall Acc:     0.9186219209989812
    Mean Acc :       0.7558250605407981
    FreqW Acc :      0.8519259882817697
    Mean IoU :       0.669605698908336
    Overall F1:      0.7700339259107747
    ----------------- Class IoU Performance ----------------
    background      : 0.8756193690879847
    skin    : 0.9108930370200288
    nose    : 0.8609366378615545
    eyeglass        : 0.6722565251628122
    left_eye        : 0.7592426552810816
    right_eye       : 0.751199043128144
    left_brow       : 0.7068159169124701
    right_brow      : 0.7019964572196532
    left_ear        : 0.7299301970980586
    right_ear       : 0.6945500409690668
    mouth   : 0.7839809881539818
    upper_lip       : 0.7214905786735428
    lower_lip       : 0.7862755127840406
    hair    : 0.8597730563739198
    hat     : 0.2275459405193996
    earring : 0.2749714537612157
    necklace        : 0.006243384910034776
    neck    : 0.7872760306285664
    cloth   : 0.6115114537128281
    --------------------------------------------------------

    ```

    </details>

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[gpen-parsenet-vdsp_params.json](../build_in/vdsp_params/gpen-parsenet-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_parsenet_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/gpen-parsenet-vdsp_params.json \
        -i 2 p 2 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path /path/to/CelebAMask-HQ/test_img \
        --target_path  /path/to/CelebAMask-HQ/test_img_npz \
        --text_path npz_datalist.txt
    ```
    
    - vamp推理得到npz结果
    ```bash
    vamp -m deploy_weights/official_parsenet_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/gpen-parsenet-vdsp_params.json \
        -i 2 p 2 -b 1 \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz文件并统计精度，参考：[vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)
    ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
        --src_dir /path/to/CelebAMask-HQ/test_img \
        --gt_dir /path/to/CelebAMask-HQ/test_label \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir ./npz_output \
        --input_shape 512 512 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```

