
## zllrunning版本

### step.1 获取预训练模型
```
link: https://github.com/zllrunning/face-parsing.PyTorch
branch: master
commit: d2e684cf1588b46145635e8fe7bcc29544e5537e
```

一般在原始仓库内进行模型转为onnx或torchscript。在原仓库test或val脚本内，如[test.py](./face_parsing/test.py)，定义模型和加载训练权重后，添加以下脚本可实现：

```python
checkpoint = save_pth

input_shape = (1, 3, 512, 512)
shape_dict = [("input", input_shape)]
input_data = torch.randn(input_shape)

scripted_model = torch.jit.trace(net, input_data).eval()
scripted_model.save(checkpoint.replace(".pth", "-512.torchscript.pt"))
scripted_model = torch.jit.load(checkpoint.replace(".pth", "-512.torchscript.pt"))

# # onnx==10.0.0，opset 10
# import onnx
# torch.onnx.export(net, input_data, checkpoint.replace(".pth", "-512.onnx"), input_names=["input"], output_names=["output"], opset_version=11)
# shape_dict = {"input": input_shape}
# onnx_model = onnx.load(checkpoint.replace(".pth", "-512.onnx"))
```

> **Note**:
> 
> onnx未能导出，有不支持算子：`RuntimeError: Failed to export an ONNX attribute 'onnx::Gather'`
>

### step.2 准备数据集
- 下载[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)数据集，解压
- 这个仓库的标签和数据集官方标签顺序不一样，需要用仓库脚本[prepropess_data.py](./face_parsing/prepropess_data.py)（参考自[源仓库](https://github.com/zllrunning/face-parsing.PyTorch/blob/master/prepropess_data.py)生成验证数据集）

### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[zllrunning_config.yaml](../vacc_code/build/zllrunning_config.yaml)
3. 执行转换

   ```bash
    vamc build ../vacc_code/build/zllrunning_config.yaml
   ```
### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path CelebAMask-HQ/bisegnet_test_img \
    --target_path  CelebAMask-HQ/bisegnet_test_img_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[zllrunning-bisenet-vdsp_params.json](../vacc_code/vdsp_params/zllrunning-bisenet-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/bisenet-int8-kl_divergence-3_512_512-vacc/bisenet \
    --vdsp_params ../vacc_code/vdsp_params/zllrunning-bisenet-vdsp_params.json \
    -i 2 p 2 -b 1
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/bisenet-int8-kl_divergence-3_512_512-vacc/bisenet \
    --vdsp_params vacc_code/vdsp_params/zllrunning-bisenet-vdsp_params.json \
    -i 2 p 2 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [zllrunning_vamp_eval.py](../vacc_code/vdsp_params/zllrunning_vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/zllrunning_vamp_eval.py \
    --src_dir CelebAMask-HQ/bisegnet_test_img \
    --gt_dir CelebAMask-HQ/bisegnet_test_mask \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 512 512 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


#### Tips

- 默认模型有三个输出，[model.py#L254](https://github.com/zllrunning/face-parsing.PyTorch/blob/master/model.py#L254)，只返回`feat_out32`一个输出，可改善vaststream run速度（提升近1倍，bs1-单进程-AI 16%-33fps ---> 60fps）（[pretrained_weights.json](../pretrained_weights.json)内提供的为单输出）
- 
    <details><summary>eval metrics</summary>

    ```
    torch 512 classes = 19
    ----------------- Total Performance --------------------
    Overall Acc:     0.9556352229294156
    Mean Acc :       0.8337249435210217
    FreqW Acc :      0.9164882022332891
    Mean IoU :       0.7426032880777604
    Overall F1:      0.8410596827641568
    ----------------- Class IoU Performance ----------------
    background      : 0.9378292772392331
    skin    : 0.9284966349726388
    nose    : 0.6149654277845954
    eyeglass        : 0.601570945020349
    left_eye        : 0.6525746650178532
    right_eye       : 0.6476684443446906
    left_brow       : 0.8319301832614219
    right_brow      : 0.6705880061787682
    left_ear        : 0.6571437967301036
    right_ear       : 0.4255793179986799
    mouth   : 0.8700320964779847
    upper_lip       : 0.8356214172588405
    lower_lip       : 0.7608609449202323
    hair    : 0.7992254674335968
    hat     : 0.8605482739767997
    earring : 0.3527505833196731
    necklace        : 0.8263593822900838
    neck    : 0.9298832924238305
    cloth   : 0.9058343168280723


    vacc 512 fp16
    ----------------- Total Performance --------------------
    Overall Acc:     0.9556578441703113
    Mean Acc :       0.833372987499022
    FreqW Acc :      0.916523859125999
    Mean IoU :       0.742535004231554
    Overall F1:      0.8410015127154231
    ----------------- Class IoU Performance ----------------
    background      : 0.9378759218073169
    skin    : 0.9285489511457389
    nose    : 0.6146148041426784
    eyeglass        : 0.6010793757831302
    left_eye        : 0.6523158728168997
    right_eye       : 0.6473657791416703
    left_brow       : 0.8320022436292658
    right_brow      : 0.6705693149717842
    left_ear        : 0.6571108468897708
    right_ear       : 0.42551992575688297
    mouth   : 0.8703116341680609
    upper_lip       : 0.8355227523616399
    lower_lip       : 0.7607766298012781
    hair    : 0.7993262178815398
    hat     : 0.8605150492840775
    earring : 0.3526342243226829
    necklace        : 0.826320220512025
    neck    : 0.9299156756029198
    cloth   : 0.9058396403801641
    --------------------------------------------------------


    vacc 512 int8 kl
    ----------------- Total Performance --------------------
    Overall Acc:     0.9556897452579036
    Mean Acc :       0.8323962754949473
    FreqW Acc :      0.9166093566148232
    Mean IoU :       0.7413797567604206
    Overall F1:      0.8401734007068263
    ----------------- Class IoU Performance ----------------
    background      : 0.9381407091955711
    skin    : 0.9287122321236828
    nose    : 0.6082316054863004
    eyeglass        : 0.6015556332608095
    left_eye        : 0.6488778693748087
    right_eye       : 0.6450650386752117
    left_brow       : 0.8322012724309388
    right_brow      : 0.6648368442550666
    left_ear        : 0.6505651232149232
    right_ear       : 0.42300831811896916
    mouth   : 0.8705586203375313
    upper_lip       : 0.8353515645581359
    lower_lip       : 0.7602545709801846
    hair    : 0.8000063660197961
    hat     : 0.8603837252231518
    earring : 0.356220163365939
    necklace        : 0.8267675595800013
    neck    : 0.9300418608525668
    cloth   : 0.9054363013944026
    ```
    </details>
