## pytorch

```
# pytorch
link: https://github.com/czczup/FAST/tree/main
branch: main
commit: 7ab105474ab226ab74a346b0648d2e2baab67d79
```

### step.1 获取预训练模型

1. 从项目中下载torch模型

2. 修改代码，生成onnx模型格式
   
   - 修改`test.py`，参考[test.py](../source_code/fast/test.py)
   - 修改模型定义文件`models/fast.py`，参考[fast.py](../source_code/fast/fast.py)

3. 运行脚本

   ```bash
   python test.py config/fast/ctw/fast_tiny_ctw_512_finetune_ic17mlt.py weights/fast_tiny_ctw_512_finetune_ic17mlt.pth
   ```

### step.2 准备数据集
- 下载[CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)数据集，通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [pytorch_fast.yaml](../build_in/build/pytorch_fast.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd fast
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/pytorch_fast.yaml
    ```

### step.4 模型推理

- 参考：[fast_vsx.py](../build_in/vsx/python/fast_vsx.py)
    ```bash
    python ../build_in/vsx/python/fast_vsx.py \
        --file_path  /path/to/ctw1500/test/text_images  \
        --model_prefix_path deploy_weights/pytorch_fast_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/pytorch-fast_tiny_ctw_512_finetune_ic17mlt-vdsp_params.json \
        --save_dir ./infer_output \
        --device 0
    ```

- 参考[eval.py](../source_code/eval/script.py)，对上述结果压缩后进行预测
    ```bash
    cd ./infer_output
    zip -r vsx_pred.zip *
    mv vsx_pred.zip ../../source_code/eval/
    cd ../../source_code/eval/
    python script.py -g=ctw1500-gt.zip -s=vsx_pred.zip
    ```


    ```
    # fp16
    num_gt, num_det:  3068 2599
    Origin:
    recall:  0.7327 precision:  0.8649 hmean:  0.7934
    TIoU-metric:
    tiouRecall: 0.511 tiouPrecision: 0.0 tiouHmean: 0.0

    # int8
    num_gt, num_det:  3068 2596
    Origin:
    recall:  0.7337 precision:  0.8671 hmean:  0.7948
    TIoU-metric:
    tiouRecall: 0.512 tiouPrecision: 0.0 tiouHmean: 0.0

    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[pytorch-fast_base_ctw_512_finetune_ic17mlt-vdsp_params.json](../build_in/vdsp_params/pytorch-fast_base_ctw_512_finetune_ic17mlt-vdsp_params.json)
    ```
    vamp -m ./deploy_weights/pytorch_fast_int8/mod --vdsp_params ../build_in/vdsp_params/pytorch-fast_tiny_ctw_512_finetune_ic17mlt-vdsp_params.json -i 1 p 1 -b 1
    ```


2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py --dataset_path /path/to/ocr/ctw1500/test_images/ --target_path /path/to/ocr/ctw1500/test_images_npz --text_path npz_datalist.txt
    ```
    
    - vamp推理得到npz结果：
    ```bash
    vamp -m ./deploy_weights/pytorch_fast_int8/mod --vdsp_params ../build_in/vdsp_params/pytorch-fast_tiny_ctw_512_finetune_ic17mlt-vdsp_params.json -i 1 p 1 -b 1 --datalist npz_datalist.txt --path_output npz_output
    ```
    
    - 解析npz结果，参考：[npz_decode.py](../build_in/vdsp_params/npz_decode.py)，
    ```bash
    python ../build_in/vdsp_params/npz_decode.py \
        --gt_dir /path/to//ocr/ctw1500/test_images/ \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output 
    ```

    - 统计精度结果
    ```bash
    cd vsx_int8_pred
    zip -r vsx_int8_pred.zip *
    mv vsx_int8_pred.zip ../../source_code/eval/
    cd ../../source_code/eval/
    python script.py -g=ctw1500-gt.zip -s=vsx_int8_pred.zip
    ```
