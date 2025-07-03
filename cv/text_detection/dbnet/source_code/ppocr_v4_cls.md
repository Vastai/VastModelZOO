## PPOCR CLS
文本方向分类器主要用于图片非0度的场景下，在这种场景下需要对图片里检测到的文本行进行一个转正的操作。在PaddleOCR系统内， 文字检测之后得到的文本行图片经过仿射变换之后送入识别模型，此时只需要对文字进行一个0和180度的角度分类，因此PaddleOCR内置的 文本方向分类器只支持了0和180度的分类

```
link: https://github.com/PaddlePaddle/PaddleOCR
branch: release/2.7
commit: https://github.com/PaddlePaddle/PaddleOCR.git
```
- [PP-OCRv4介绍](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/PP-OCRv4_introduction.md)
- [方向分类器介绍](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/angle_class.md)

### step.1 模型准备
1. 首先，需要进入到PaddleOCR工程主目录，安装PaddleOCR：
```
pip install -r requrement.txt
python setup.py install
```
2. 下载推理模型：
```shell
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
mkdir -p weights/
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar -C weights/
```
3. 将推理模型转换为onnx：

```shell
paddle2onnx --model_dir weights/ch_ppocr_mobile_v2.0_cls_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file weights/ch_ppocr_mobile_v2.0_cls_infer/cls_model.onnx \
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

### step.2 准备数据集
- 因为该方向模型是专用模型且结构简单，官方并未给出专门的测试集。这里参考官方方案，使用6张测试图进行测试即可。
- 下载[测试图](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.7/doc/imgs_words/ch)到本地cls_test文件夹
    - 需要使用画图工具将word_1.jpg文件顺时针旋转180度，得到word_6.jpg，放入cls_test文件夹
    - cls_test文件夹的内容如下
    ```
    cls_test/
    ├── word_1.jpg
    ├── word_2.jpg
    ├── word_3.jpg
    ├── word_4.jpg
    ├── word_5.jpg
    └── word_6.jpg
    ```
- 当前使用的6张图片，使用paddleocr的官方测试结果如下：
```
word_1.jpg result: ('0', 0.9998784)
word_2.jpg result: ('0', 1.0)
word_3.jpg result: ('0', 1.0)
word_4.jpg result: ('0', 0.9999982)
word_5.jpg result: ('0', 0.9999988)
word_6.jpg result: ('180', 0.9999759)
```

### step.3 模型转换
1. 根据具体模型修改配置文件
    -[ppocr_v4_cls.yaml](../build_in/build/ppocr_v4_cls.yaml)
    
2. 模型编译
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd dbnet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ppocr_v4_cls.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights/ppocr_v4_cls_run_stream_int8`文件夹，其中包含转换后的模型文件。

### step.4 模型推理
1. 参考[cls_vsx.py](../build_in/vsx/python/ppocr_v4_cls_vsx.py)，进行vsx推理和eval评估
    ```bash
    python ../build_in/vsx/python/ppocr_v4_cls_vsx.py \
        --file_path ../build_in/data \
        --model_prefix_path deploy_weights/ppocr_v4_cls_run_stream_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/ppocr-ch_PP_OCRv4_cls-vdsp_params.json \
        --output_file ppocrv4_cls_runstream_pred.txt
    ```
    - 注意替换命令行中--file_path和--label_txt为实际路径
    - 输出结果就在ppocrv4_cls_runstream_pred.txt中，结果如下：
    ```
    word_1 [('0', 1.0)]
    word_2 [('0', 1.0)]
    word_3 [('0', 1.0)]
    word_4 [('0', 1.0)]
    word_5 [('0', 1.0)]
    word_6 [('180', 1.0)]
    ```

### step.5 性能测试
1. 性能测试，配置vdsp参数[ppocr-ch_PP_OCRv4_cls-vdsp_params.json](../build_in/vdsp_params/ppocr-ch_PP_OCRv4_cls-vdsp_params.json)，执行：
    - 由于vamp暂不支持该性能测试，所以这里使用python脚本进行性能测试
    # 测试最大吞吐
    ```bash
    python3 ../build_in/vsx/python/ppocr_v4_cls_prof.py \
        -m deploy_weights/ppocr_v4_cls_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/ppocr-ch_PP_OCRv4_cls-vdsp_params.json \
        --device_ids [0] \
        --batch_size 8 \
        --instance 1 \
        --shape "[3,48,192]" \
        --iterations 200 \
        --percentiles "[50,90,95,99]" \
        --input_host 1 \
        --queue_size 1
    ```
    # 测试最小时延
    ```bash
    python3 ../build_in/vsx/python/ppocr_v4_cls_prof.py \
    -m deploy_weights/ppocr_v4_cls_run_stream_int8/mod \
    --vdsp_params ../build_in/vdsp_params/ppocr-ch_PP_OCRv4_cls-vdsp_params.json \
    --device_ids [0] \
    --batch_size 1 \
    --instance 1 \
    --shape "[3,48,192]" \
    --iterations 500 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    ```
    
