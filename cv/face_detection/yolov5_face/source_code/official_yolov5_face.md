### step.1 获取预训练模型

基于原始库提供的脚本[export.py](https://github.com/deepcam-cn/yolov5-face/blob/master/export.py)可以将原模型转为onnx或torchscript格式，需修改[yolo.py](https://github.com/deepcam-cn/yolov5-face/blob/master/models/yolo.py)中`Detect`类中`Forward`函数，使得转出的模型不包含后处理，如下

```python
def forward(self, x):
    # x = x.copy()  # for profiling
    z = []  # inference output
    if True:
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
        return x

```

### step.2 准备数据集
- [校准数据集](https://huggingface.co/datasets/wider_face/blob/main/data/WIDER_val.zip)
- [评估数据集](https://huggingface.co/datasets/wider_face/blob/main/data/WIDER_val.zip)


### step.3 模型转换
1. 根据具体模型修改配置文件
    - [official_yolov5_face.yaml](../build_in/build/official_yolov5_face.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 执行转换
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_yolov5_face.yaml
    ```

### step.4 模型推理
1. 参考[vsx脚本](../build_in/vsx/python/vsx.py)，修改参数并运行如下脚本
    ```bash
    python ../build_in/vsx/python/vsx.py \
        --file_path  /path/to/widerface/val/images \
        --model_prefix_path deploy_weights/official_yolov5_face_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/pytorch-yolov5s_face-vdsp_params.json \
        --save_dir ./infer_output \
        --device 0
    ```
    - 注意替换命令行中--file_path为实际路径

2. [evaluation.py](../../common/eval/evaluation.py)，精度统计，指定gt路径和上步骤中的txt保存路径，即可获得精度指标
    - 注意需要先执行以下命令安装依赖库（无须重复安装）
    ```
    cd ../../common/eval/;
    python3 setup.py build_ext --inplace;
    ```
    - 然后切换到之前的workspace目录进行精度验证
    ```bash
    python ../../common/eval/evaluation.py -p infer_output/ -g ../../common/eval/ground_truth
    ```
    - 测试精度如下：
    ```
    ==================== Results ====================
    Easy   Val AP: 0.9366730285439512
    Medium Val AP: 0.9203239690788733
    Hard   Val AP: 0.8319852517254902
    =================================================

    ```

### step.5 性能精度测试

1. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`widerface_npz_list.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path /path/to/widerface/val/images --target_path  /path/to/widerface_npz --text_path widerface_npz_list.txt
    ```

2. 性能测试
    ```bash
    vamp -m deploy_weights/official_yolov5_face_int8/mod  --vdsp_params ../build_in/vdsp_params/pytorch-yolov5s_face-vdsp_params.json -i 2 p 2 -b 1
    ```

3. 精度测试
   ```bash
    vamp -m deploy_weights/official_yolov5_face_int8/mod  --vdsp_params ../build_in/vdsp_params/pytorch-yolov5s_face-vdsp_params.json -i 2 p 2 -b 1 --datalist widerface_npz_list.txt --path_output vamp_result
    ```

4. 解析输出结果用于精度评估
   参考[npz_decode.py](./build_in/vdsp_params/npz_decode.py)将输出结果进行解析并进行保存
   ```bash
   python npz_decode.py --txt result_npz --input_image_dir /path/to/widerface/val/images --model_size 640 640 --vamp_datalist_path widerface_npz_list.txt --vamp_output_dir vamp_result
   ```

5. [evaluation.py](../../common/eval/evaluation.py)，精度统计，指定上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    # 安装评估包
    cd ../../common/eval/
    python setup.py build_ext --inplace

    # 切换到之前的workspace目录进行精度验证
    python ../../common/eval/evaluation.py -p result_npz -g ../../common/eval/ground_truth
   ```