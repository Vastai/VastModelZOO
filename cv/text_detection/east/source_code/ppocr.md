## PPOCR

```
link: https://github.com/PaddlePaddle/PaddleOCR
branch: v2.7
commit: b17c2f3a5687186caca590a343556355faacb243
```

### step.1 获取预训练模型
首先，ppocr下载的是训练模型，需要转换为推理模型。在ppocr仓库目录内，执行：

```shell
python3 tools/export_model.py -c configs/det/det_r50_vd_east.yml  -o Global.pretrained_model=./models/detection/EAST/det_r50_vd_east_v2.0_train//best_accuracy  Global.save_inference_dir=./models/detection/EAST/inference
```

然后，将推理模型转换为onnx：

```shell
paddle2onnx --model_dir ./models/detection/EAST/inference --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ./models/detection/EAST/det_r50_vd_east.onnx --opset_version 11
```

### step.2 准备数据集
- 下载[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)数据集，通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件


### step.3 模型转换
1. 根据具体模型修改配置文件
    - [ppocr_east.yaml](../build_in/build/ppocr_east.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd east
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ppocr_east.yaml
    ```

### step.4 模型推理
1. 参考[east_vsx.py](../build_in/vsx/python/east_vsx.py)，进行vsx推理和eval评估
    ```bash
    python ../build_in/vsx/python/east_vsx.py \
        --file_path path/to/icdar2015/Challenge4/ch4_test_images \
        --model_prefix_path deploy_weights/ppocr_east_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/ppocr-det_mv3_east-vdsp_params.json \
        --label_txt /path/to/icdar2015/Challenge4/test_icdar2015_label.txt
    ```
    - 注意替换命令行中--file_path和--label_txt为实际路径 
    - 精度数据就在打印信息中，如下：
    ```
    metric:  {'precision': 0.7695, 'recall': 0.7409725565719788, 'hmean': 0.7549668874172185}
    ```


### step.5 性能精度测试
1. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path path/to/ch4_test_images \
    --target_path path/to/ch4_test_images_npz \
    --text_path npz_datalist.txt
    ```

2. 性能测试，配置vdsp参数[ppocr-det_r50_vd_east-vdsp_params.json](../build_in/vdsp_params/ppocr-det_r50_vd_east-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/ppocr_east_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/ppocr-det_r50_vd_east-vdsp_params.json -i 1 p 1 -b 1 -s 3,704,1280
    ```

3. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/det_r50_vd_east-int8-kl_divergence-3_704_1280-vacc/det_r50_vd_east \
    --vdsp_params ../build_in/vdsp_params/ppocr-det_r50_vd_east-vdsp_params.json \
    -i 1 p 1 -b 1 -s 3,704,1280 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```

4. [vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
    --gt_dir path/to/icdar2015/Challenge4/ch4_test_images \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir npz_output \
    --input_shape 704 1280 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```