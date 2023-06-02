
## GPEN版本

### step.1 获取预训练模型
```
link: https://github.com/yangxy/GPEN/blob/main/face_parse/parse_model.py
branch: master
commit: c9cc29009b633788a77d782ba102cee913e3a349
```
- 克隆源仓库
- 基于[export_onnx.py](./gpen/face_parse/export_onnx.py)，转换模型


### step.2 准备数据集
- 下载[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)数据集，按官方处理数据集

### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[config.yaml](../vacc_code/build/config.yaml)
3. 执行转换

   ```bash
    vamc build ../vacc_code/build/config.yaml
   ```

### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path CelebAMask-HQ/test_img \
    --target_path  CelebAMask-HQ/test_img_npz \
    --text_path test_npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[gpen-parsenet-vdsp_params.json](../vacc_code/vdsp_params/gpen-parsenet-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/parsenet-int8-kl_divergence-3_512_512-vacc/parsenet \
    --vdsp_params ../vacc_code/vdsp_params/gpen-parsenet-vdsp_params.json \
    -i 2 p 2 -b 1
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/parsenet-int8-kl_divergence-3_512_512-vacc/parsenet \
    --vdsp_params vacc_code/vdsp_params/gpen-parsenet-vdsp_params.json \
    -i 2 p 2 -b 1 \
    --datalist test_npz_datalist.txt \
    --path_output npz_output
    ```
5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --src_dir CelebAMask-HQ/test_img \
    --gt_dir CelebAMask-HQ/test_label \
    --input_npz_path test_npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 512 512 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


#### Tips
- 源仓库没有提供训练代码
- 
    <details><summary>eval metrics</summary>

    ```
    torch 512 classes = 19
    ----------------- Total Performance --------------------
    Overall Acc:     0.9185948817655655
    Mean Acc :       0.7556176746922927
    FreqW Acc :      0.8518606947242394
    Mean IoU :       0.6694148784296005
    Overall F1:      0.7698035137732336
    ----------------- Class IoU Performance ----------------
    background      : 0.875538068267407
    skin    : 0.9108304414489041
    nose    : 0.8610505174981719
    eyeglass        : 0.6719081391650369
    left_eye        : 0.7591563134743745
    right_eye       : 0.750934472799956
    left_brow       : 0.7067620299090169
    right_brow      : 0.7019382979861717
    left_ear        : 0.7299365483950949
    right_ear       : 0.6948633190773966
    mouth   : 0.7839446282449941
    upper_lip       : 0.7211721211990042
    lower_lip       : 0.7862414264537255
    hair    : 0.8596910585830915
    hat     : 0.22707464163767346
    earring : 0.272853673270839
    necklace        : 0.005866457406481581
    neck    : 0.7872765218126411
    cloth   : 0.6118440135324256


    parsenet-fp16-none-3_512_512-debug
    ----------------- Total Performance --------------------
    Overall Acc:     0.8817774472763451
    Mean Acc :       0.7161369193171488
    FreqW Acc :      0.788360389866883
    Mean IoU :       0.6288877411754571
    Overall F1:      0.7397868565689768
    ----------------- Class IoU Performance ----------------
    background      : 0.8081547420247954
    skin    : 0.8692771486857429
    nose    : 0.8530692647388858
    eyeglass        : 0.536301156734449
    left_eye        : 0.7575150879579671
    right_eye       : 0.7498317290108022
    left_brow       : 0.706394078670431
    right_brow      : 0.701544185130455
    left_ear        : 0.7169197740456301
    right_ear       : 0.6769096935972514
    mouth   : 0.7796111219905858
    upper_lip       : 0.7090610344181969
    lower_lip       : 0.7781928110932722
    hair    : 0.7992159953110064
    hat     : 0.21106679170724543
    earring : 0.24767920593030526
    necklace        : 0.016606347256882294
    neck    : 0.6082240208058441
    cloth   : 0.42329289322393615
    --------------------------------------------------------


    parsenet-int8-kl_divergence-3_512_512-debug
    ----------------- Total Performance --------------------
    Overall Acc:     0.8821907138013975
    Mean Acc :       0.7078018716935736
    FreqW Acc :      0.7884629667980032
    Mean IoU :       0.6272120370442974
    Overall F1:      0.7371717241152979
    ----------------- Class IoU Performance ----------------
    background      : 0.8094216254782843
    skin    : 0.8702658007455422
    nose    : 0.8483825524255447
    eyeglass        : 0.5432061466596172
    left_eye        : 0.7563790920375586
    right_eye       : 0.7562817924847844
    left_brow       : 0.7012892053219268
    right_brow      : 0.6986628186572649
    left_ear        : 0.7153607946457127
    right_ear       : 0.6735261252658867
    mouth   : 0.7818841442206066
    upper_lip       : 0.7089328809805324
    lower_lip       : 0.775382928048774
    hair    : 0.7972812442705548
    hat     : 0.20365213827833115
    earring : 0.23207659838270017
    necklace        : 0.0026841145047185287
    neck    : 0.6072135081379337
    cloth   : 0.43514519329537643
    --------------------------------------------------------
    ```
    </details>