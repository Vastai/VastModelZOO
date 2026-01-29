### step.1 模型准备
```
link: https://github.com/timesler/facenet-pytorch
branch: master
commit: 555aa4bec20ca3e7c2ead14e7e39d5bbce203e4b
```

- 克隆仓库，将[export.py](./pytorch/export.py)移动至{facenet-pytorch}工程目录，转换获得onnx和torchscript权重
    ```
    conda create -n facenet python=3.10
    conda activate facenet
    # pip install facenet-pytorch
    git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch
    cd facenet_pytorch
    python setup.py develop
    pip install pandas onnx onnxsim
    
    # facenet-pytorch仓库将自动下载原始权重，如网络较慢，可手动下载
    # wget https://gh-proxy.org/https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt
    # wget https://gh-proxy.org/https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt
    # cp 20180402-114759-vggface2.pt ~/.cache/torch/checkpoints
    # cp 20180408-102900-casia-webface.pt ~/.cache/torch/checkpoints
    
    python facenet_pytorch/export.py
    ```


### step.2 准备数据集

- [校准数据集](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
- [评估数据集](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw)
- [标签pairs.txt](https://github.com/davidsandberg/facenet/blob/master/data/pairs.txt)  
- 通过[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，并生成npz_datalist.txt文件


### step.3 模型转换
1. 根据具体模型,修改编译配置文件
    - [config.yaml](../build_in/build/pytorch_facenet.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/pytorch_facenet.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights/pytorch_facenet_int8`文件夹，其中包含转换后的模型文件。

### step.4 模型推理
1. 参考[vsx脚本](../build_in/vsx/python/vsx_inference.py)，修改参数并运行如下脚本
    ```bash
    # pip install scipy==1.9.1
    python ../build_in/vsx/python/vsx_inference.py \
        --image_dir  /path/to/lfw_mtcnnpy_160 \
        --model_prefix_path deploy_weights/pytorch_facenet_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/pytorch-facenet_vggface2-vdsp_params.json \
        --lfw_pairs /path/to/pairs.txt \
        --device 0
    ```
    - 注意替换命令行中--file_path为实际路径
    - 精度信息就在打印信息中，如下：
    ```
    Accuracy: 0.98900+-0.00611
    Validation rate: 0.94967+-0.01602 @ FAR=0.00100
    Area Under Curve (AUC): 0.99869
    Equal Error Rate (EER): 0.01227
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
    vamp -m deploy_weights/pytorch_facenet_int8/mod --vdsp_params ../build_in/vdsp_params/pytorch-facenet_vggface2-vdsp_params.json -i 1 p 1 -b 1
    ```

3. 精度测试，输出npz结果
    ```bash
    vamp -m deploy_weights/pytorch_facenet_int8/mod \
    --vdsp_params ../build_in/vdsp_params/pytorch-facenet_vggface2-vdsp_params.json \
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


- <details><summary>精度</summary>

    ```
    # lfw_mtcnnpy_160
    facenet_vggface2-int8-percentile-1_3_160_160-vacc
    Accuracy: 0.99350+-0.00411
    Validation rate: 0.97633+-0.01269 @ FAR=0.00067
    Area Under Curve (AUC): 0.99959
    Equal Error Rate (EER): 0.00567

    facenet_casia_webface-int8-percentile-1_3_160_160-vacc
    Accuracy: 0.98867+-0.00581
    Validation rate: 0.94900+-0.01606 @ FAR=0.00100
    Area Under Curve (AUC): 0.99869
    Equal Error Rate (EER): 0.01267
    ```
    </details>