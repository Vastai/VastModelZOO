## PPOCR DET

```
link: https://github.com/PaddlePaddle/PaddleOCR
branch: release/2.7
commit: https://github.com/PaddlePaddle/PaddleOCR.git
```
- [PP-OCRv4介绍](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/PP-OCRv4_introduction.md)

### step.1 模型准备
1. 首先，需要进入到PaddleOCR工程主目录，安装PaddleOCR：
```
pip install -r requrement.txt
python setup.py install
```
2. 下载推理模型：
```shell
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar
mkdir -p weights/
tar -xvf ch_PP-OCRv4_det_infer.tar -C weights/
```
3. 将推理模型转换为onnx：

```shell
paddle2onnx --model_dir weights/ch_PP-OCRv4_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file weights/ch_PP-OCRv4_det_infer/inference.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True
```

> **Note**: 
- 此处需要安装paddle2onnx-0.9.7版本
- 安装paddle2onnx-0.9.7版本命令为：
```
pip install paddle2onnx==0.9.7 --no-cache-dir
```

- 文本检测结束后，还需要使用方向分类网络对文本进行分类，判断文本的方向（正向或倒向）
- [方向分类模型](./ppocr_v4_cls.md)

### step.2 准备数据集
- 下载[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)数据集
  - 测试图像：[ch4_test_images](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=4&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy8/Y29tPWRvd25sb2FkcyZhY3Rpb249ZG93bmxvYWQmZmlsZT1jaDRfdGVzdF9pbWFnZXMuemlw)
  - 测试图像标签：[test_icdar2015_label.txt](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=4&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9kb3dubG9hZHMvQ2hhbGxlbmdlNF9UZXN0X1Rhc2sxX0dULnppcA==)
    - 需要将下载的官网`label`转换支持的数据格式`test_icdar2015_label.txt`,具体转换方式可参考[测评数据集说明](../README.md)
  - 通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件


### step.3 模型转换
1. 根据具体模型修改配置文件
    -[ppocr_v4_dbnet.yaml](../build_in/build/ppocr_v4_dbnet.yaml)

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd dbnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ppocr_v4_dbnet.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights/ppocr_v4_dbnet_run_stream_int8`文件夹，其中包含转换后的模型文件。

### step.4 模型推理
1. 参考[dbnet_vsx.py](../build_in/vsx/python/dbnet_vsx.py)，进行vsx推理和eval评估
    ```bash
    python ../build_in/vsx/python/dbnet_vsx.py \
        --file_path path/to/icdar2015/Challenge4/ch4_test_images \
        --model_prefix_path deploy_weights/ppocr_v4_dbnet_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/ppocr-ch_PP_OCRv4_det-vdsp_params.json \
        --label_txt /path/to/icdar2015/Challenge4/test_icdar2015_label.txt
    ```
    - 注意替换命令行中--file_path和--label_txt为实际路径
    - 精度信息就在打印信息中，结果如下：
    ```
    metric:  {'precision': 0.5401785714285714, 'recall': 0.4077997111218103, 'hmean': 0.46474622770919066}
    ```
    - PPOCR v4系列模型并不是在ICDAR2015数据集下训练，因此精度偏低


### step.5 性能精度测试
1. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path ch4_test_images \
    --target_path ch4_test_images_npz \
    --text_path npz_datalist.txt
    ```

2. 性能测试，配置vdsp参数[ppocr-ch_PP_OCRv4_det-vdsp_params.json](../build_in/vdsp_params/ppocr-ch_PP_OCRv4_det-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/ppocr_v4_dbnet_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/ppocr-ch_PP_OCRv4_det-vdsp_params.json -i 1 p 1 -b 1 -s [3,736,1280]
    ```
    
3. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/ppocr_v4_dbnet_run_stream_int8/mod \
    --vdsp_params ../build_in/vdsp_params/ppocr-ch_PP_OCRv4_det-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,736,1280] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```

4. [vamp_eval.py](../build_in/vdsp_params/vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../build_in/vdsp_params/vamp_eval.py \
    --gt_dir icdar2015/Challenge4/ch4_test_images \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir outputs/dbnet \
    --input_shape 736 1280 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```