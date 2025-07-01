## PPOCR

```
link: https://github.com/PaddlePaddle/PaddleOCR
branch: v2.6.0
commit: 56aaead6d06b9ef6c6ecb8655f5a571f579f939e
```

### step.1 获取预训练模型
首先，ppocr下载的是训练模型，需要转换为推理模型。在ppocr仓库目录内，执行：

```shell
python tools/export_model.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model=weights/det_mv3_db_v2.0_train/best_accuracy Global.save_inference_dir=weights/det_mv3_db_v2.0_train_inference/

python tools/export_model.py -c configs/det/det_r50_vd_db.yml -o Global.pretrained_model=weights/det_r50_vd_db_v2.0_inference/best_accuracy Global.save_inference_dir=weights/det_r50_vd_db_v2.0_inference/
```

然后，将推理模型转换为onnx：

```shell
paddle2onnx --model_dir weights/det_r50_vd_db_v2.0_train_inference \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file weights/det_r50_vd_db_v2.0_train_inference/inference.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True
```

> **Note**: 

- 此处需要安装paddle2onnx-0.9.7版本，paddle2onnx-1.0.0版本在量化build时会前端报错：`TVMError: Check failed: type_code_ == kDLFloat (8 vs. 2) : expected float but get ObjectCell`，和[OP#13130](http://openproject.vastai.com/projects/model-zoo-and-benchmark-of-sz/work_packages/13130/activity)报错一致。

- DB++_ResNet50模型无法转换为onnx，paddle2onnx转换时提示里面有dcn算子不支持


### step.2 准备数据集
- 下载[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)数据集，通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件
- 处理好的数据集：
  - 测试图像：[ch4_test_images](http://192.168.20.139:8888/vastml/dataset/ocr/icdar2015/Challenge4/ch4_test_images/?download=zip)
  - 测试图像npz：[ch4_test_images_npz](http://192.168.20.139:8888/vastml/dataset/ocr/icdar2015/Challenge4/ch4_test_images_npz/?download=zip)
  - 测试图像标签：[test_icdar2015_label.txt](http://192.168.20.139:8888/vastml/dataset/ocr/icdar2015/Challenge4/test_icdar2015_label.txt)
  - 测试图像npz_datalist.txt：[npz_datalist.txt](http://192.168.20.139:8888/vastml/dataset/ocr/icdar2015/Challenge4/npz_datalist.txt)

### step.3 模型转换
1. 获取[vamc](../../../docs/doc_vamc.md)模型转换工具
2. 根据具体模型修改配置文件，[ppocr_dbnet.yaml](../vacc_code/build/ppocr_dbnet.yaml)
3. 执行转换
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd dbnet
    mkdir workspace
    cd workspace
    vamc compile ../vacc_code/build/ppocr_dbnet.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights/ppocr_dbnet_run_stream_int8`文件夹，其中包含转换后的模型文件。

### step.4 模型推理
> **runmodel**形式，注意编译参数backend.type: tvm_runmodel
1. 参考[runmodel](../vacc_code/runmodel/ppocr_sample_runmodel.py)，修改启动参数和预处理相关参数并运行如下脚本，进行runmodel推理和eval评估
    - 修改yaml文件的内容如下，参考step.3重新转换出runmodel模型
        - `name: ppocr_dbnet_run_model_fp16`
        - `backend.type: tvm_runmodel`
        - `dtype: fp16`
    - 执行以下命令进行runmodel推理
    ```bash
    python ../vacc_code/runmodel/ppocr_sample_runmodel.py \
        --file_path /path/to/ch4_test_images \
        --model_weight_path deploy_weights/ppocr_dbnet_run_model_fp16/  \
        --model_name mod \
        --model_input_name input \
        --model_input_shape 1,3,736,1280 \
        --label_txt /path/to/test_icdar2015_label.txt \
        --save_dir ./runmodel_output 
    ```
    - 精度信息就在打印信息中，结果如下：
    ```
    metric:  {'precision': 0.8017241379310345, 'recall': 0.8059701492537313, 'hmean': 0.803841536614646}
    ```

> **runstream**形式，注意编译参数backend.type: tvm_vacc
1. 参考[dbnet_vsx.py](../vacc_code/vsx/python/dbnet_vsx.py)，进行vsx推理和eval评估
    ```bash
    python ../vacc_code/vsx/python/dbnet_vsx.py \
        --file_path path/to/icdar2015/Challenge4/ch4_test_images \
        --model_prefix_path deploy_weights/ppocr_dbnet_run_stream_int8/mod \
        --vdsp_params_info ../vacc_code/vdsp_params/ppocr-dbnet_resnet50_vd-vdsp_params.json \
        --label_txt /path/to/icdar2015/Challenge4/test_icdar2015_label.txt
    ```
    - 注意替换命令行中--file_path和--label_txt为实际路径
    - 精度信息就在打印信息中，结果如下：
    ```
    metric:  {'precision': 0.8331673306772909, 'recall': 0.8054886856042369, 'hmean': 0.8190942472460221}
    ```


### step.5 性能精度
1. 获取[vamp](../../../docs/doc_vamp.md)工具

2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path ch4_test_images \
    --target_path ch4_test_images_npz \
    --text_path npz_datalist.txt
    ```

3. 性能测试，配置vdsp参数[ppocr-dbnet_resnet50_vd-vdsp_params.json](../vacc_code/vdsp_params/ppocr-dbnet_resnet50_vd-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/ppocr_dbnet_run_stream_int8/mod --vdsp_params ../vacc_code/vdsp_params/ppocr-dbnet_resnet50_vd-vdsp_params.json -i 1 p 1 -b 1 -s [3,736,1280]
    ```
    - 性能测试数据如下：
    ```
    - number of instances in each device: 1
    devices: [0]
    batch size: 1
    samples: 1024
    forwad time (s): 9.78984
    throughput (qps): 104.598
    ai utilize (%): 95.4396
    die memory used (MB): 882.742
    e2e latency (us):
        avg latency: 161158
        min latency: 11181
        max latency: 172333
    model latency (us):
        avg latency: 9124
        min latency: 9124
        max latency: 9124
    ```
    - 本测试的硬件信息如下：
    ```
    Smi version:3.2.1
    SPI production for Bbox mode information of
    =====================================================================
    Appointed Entry:0 Device_Id:0 Die_Id:0 Die_Index:0x00000000
    ---------------------------------------------------------------------
    #               Field Name                    Value
    0              FileVersion                       V2
    1                 CardType                  VA1-16G
    2                      S/N             FCA129E00172
    3                 BboxMode              Highperf-AI
    =====================================================================
    =====================================================================
    Appointed Entry:0 Device_Id:0 Die_Id:0 Die_Index:0x00000000
    ---------------------------------------------------------------------
    OCLK:       880 MHz    ODSPCLK:    835 MHz    VCLK:       300 MHz    
    ECLK:        20 MHz    DCLK:        20 MHz    VDSPCLK:    900 MHz    
    UCLK:      1067 MHz    V3DCLK:     100 MHz    CCLK:      1000 MHz    
    XSPICLK:     50 MHz    PERCLK:     200 MHz    CEDARCLK:   500 MHz
    ```

4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/ppocr_dbnet_run_stream_int8/mod \
    --vdsp_params vacc_code/vdsp_params/ppocr-dbnet_resnet50_vd-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,736,1280] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```

5. [vamp_eval.py](../vacc_code/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/vamp_eval.py \
    --gt_dir icdar2015/Challenge4/ch4_test_images \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/dbnet \
    --input_shape 736 1280 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```

### Tips
- fp16量化后无法在VE1上跑，内存不够（736*1280）
- int8量化精度受量化数据集数量的影响很大：

    ```
    # 使用icdar2015全量验证集（icdar2015/text_localization/ch4_test_images）作为量化校准数据精度
    metric precision:0.7722772277227723
    metric recall:0.6759749638902263
    metric hmean:0.7209242618741978

    # 验证集抽取100张精度
    metric precision:0.4730100640439158
    metric recall:0.7467501203659124
    metric hmean:0.5791635548917102
    ```
- PPOCR v3系列模型并不是在ICDAR2015数据集下训练，因此精度偏低