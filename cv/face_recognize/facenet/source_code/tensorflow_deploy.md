### step.1 模型准备

```
link: https://github.com/davidsandberg/facenet
branch: master
commit: 096ed770f163957c1e56efa7feeb194773920f6e
```

- 安装tf2onnx
- 克隆仓库，将[tensorflow2onnx.py](./tensorflow/tensorflow2onnx.py)移动至{facenet}工程目录，转换获得onnx权重


### step.2 准备数据集

- [校准数据集](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
- [评估数据集](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw)
- [标签pairs.txt](https://github.com/davidsandberg/facenet/blob/master/data/pairs.txt)  
- 通过[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，并生成npz_datalist.txt文件

### step.3 模型转换

1. 根据具体模型,修改编译配置文件
    - [config.yaml](../build_in/build/tensorflow_facenet.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/tensorflow_facenet.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights/tensorflow_facenet_int8`文件夹，其中包含转换后的模型文件。

### step.4 模型推理
1. 参考[vsx脚本](../build_in/vsx/python/vsx_inference.py)，修改参数并运行如下脚本
    ```bash
    # pip install scipy==1.9.1
    python ../build_in/vsx/python/vsx_inference.py \
        --image_dir  /path/to/lfw_mtcnnpy_160 \
        --model_prefix_path deploy_weights/tensorflow_facenet_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/tensorflow-facenet_vggface2-vdsp_params.json \
        --lfw_pairs /path/to/pairs.txt \
        --device 0
    ```
    - 注意替换命令行中--file_path为实际路径
    - 精度信息就在打印信息中，如下：
    ```
    Accuracy: 0.98950+-0.00568
    Validation rate: 0.94800+-0.02001 @ FAR=0.00100
    Area Under Curve (AUC): 0.99875
    Equal Error Rate (EER): 0.01276

    ```

### step.5 性能精度测试
1. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path /path/to/lfw_mtcnnpy_160 \
    --target_path lfw_mtcnnpy_160_npz \
    --text_path npz_datalist.txt
    ```

2. 性能测试
    ```bash
    vamp -m deploy_weights/tensorflow_facenet_int8/mod --vdsp_params ../build_in/vdsp_params/tensorflow-facenet_vggface2-vdsp_params.json -i 1 p 1 -b 1
    ```

3. 精度测试，输出npz结果
    ```bash
    vamp -m eploy_weights/tensorflow_facenet_int8/mod \
    --vdsp_params ../build_in/vdsp_params/tensorflow-facenet_vggface2-vdsp_params.json \
    -i 1 p 1 -b 1 \
    --datalist npz_datalist.txt \
    --path_output vamp_result
    ```

4. [npz_decode.py](../../common/eval/npz_decode.py)，解析vamp输出的npz文件，并进行精度测试
    ```bash
    # pip install scipy==1.9.1
    python ../../common/eval/npz_decode.py \
    --gt_dir /path/to/lfw_mtcnnpy_160 \
    --gt_pairs_path /path/to/pairs.txt \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir vamp_result
    ```