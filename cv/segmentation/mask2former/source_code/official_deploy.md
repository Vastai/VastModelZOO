## Deploy

### step.1 获取模型

- 模型来源

    ```bash
    # 原始仓库
    - gitlab：https://github.com/facebookresearch/Mask2Former
    - commit: 9b0651c6c1d5b3af2e6da0589b719c514ec0d69a

    # 因Mask2Former依赖detectron2库，此处需要git clone此仓库，源码方式安装
    - gitlab：https://github.com/facebookresearch/detectron2
    - commit: 70f454304e1a38378200459dd2dbca0f0f4a5ab4

    - config: https://github.com/facebookresearch/Mask2Former/blob/main/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml
    - weight: https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl
    ```

- 环境配置
    - 安装基础环境
        - 可使用导出的conda环境安装：[mask2former.yaml](./official/mask2former.yaml)，`conda env create -f mask2former.yaml`（如有安装失败的库，第二次使用更新命令：conda env update -f mask2former.yaml）

        - 或使用以下命令安装

            ```shell
            # torch版本不一致，可能导致onnx存在差异

            conda create -n mask2former python==3.8
            conda activate mask2former

            # 安装cpu版本torch
            pip install torch==2.1.2 torchvision==0.16.2

            # 安装其它依赖
            pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
            pip3 install transformers tokenizers numpy opencv-python onnx==1.16.1 onnxsim==0.4.36 onnxruntime==1.18.1 onnx_graphsurgeon==0.5.2 decorator
            ```
    
    - 安装工具包
        ```shell
        # 安装detectron2库
        git clone git@github.com:facebookresearch/detectron2.git
        cd detectron2
        git checkout 70f454304e1a38378200459dd2dbca0f0f4a5ab4

        pip install -e .

        # 安装Mask2Former库
        git clone https://gitee.com/facebookresearch/Mask2Former.git
        cd Mask2Former
        git checkout 9b0651c6c1d5b3af2e6da0589b719c514ec0d69a

        pip install -r requirements.txt
        ```

- 以上两个工具包安装完成后，为方便vacc部署，对原始模型进行了部分修改
    - 应用git patch修改，[mask2former_git_export_onnx.patch](./official/mask2former_git_export_onnx.patch)；[detectron_git_export_onnx.patch](./official/detectron_git_export_onnx.patch)；
        ```shell
        cd /path/to/Mask2Former
        git apply mask2former_git_export_onnx.patch

        cd /path/to/Detectron2
        git apply detectron_git_export_onnx.patch
        ```

    - 导出onnx模型：

        ```bash
        cd /path/to/Mask2Former
        python demo/demo.py --config-file Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
        --input . --output . \
        MODEL.WEIGHTS /path/to/weight/model_final_3c8ec9.pkl

        # Mask2Former目录中出现了新的onnx模型`mask2former.onnx`
        ```

    - 通过onnxsim对模型进行简化

        ```bash
        onnxsim mask2former.onnx mask2former_sim.onnx
        ```

    - 修改onnx文件，替换deform_attn等自定义算子，[mask2former_set_custom.py](./official/mask2former_set_custom.py)

        ```bash
        python mask2former_set_custom.py mask2former_sim.onnx mask2former_sim_with_custom.onnx
        ```

### step.2 获取数据集
- 依据原始仓库，使用[coco val2017](https://cocodataset.org/#download)验证模型精度
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)


### step.3 模型转换
- ODSP自定义算子，需对随AI Complier包一同发布的odsp_plugin包进行编译，链接编译后路径至环境变量
    ```
    wget -O 'odsp_plugin-v1.0-20241231-100-linux-x86_64.tar.gz'  http://devops.vastai.com/kapis/artifact.kubesphere.io/v1alpha1/artifact?artifactid=4410

    mkdir odsp_plugin
    tar -xzvf odsp_plugin-v1.0-20241231-100-linux-x86_64.tar.gz -C odsp_plugin

    cd odsp_plugin/vastai
    sudo ./build.sh

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/odsp_plugin/vastai/lib:/path/to/odsp_plugin/protobuf/lib/x86_64
    ```
- 根据具体模型修改配置文件
    - [official_build.yaml](../build_in/build/official_build.yaml)
    - 注意当前只支持FP16
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 模型编译
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd mask2former
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_build.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights/mask2former_run_stream_fp16`文件夹，其中包含转换后的模型文件。

### step.4 模型推理
1. 参考[vsx_inference.py](../build_in/vsx/vsx_inference.py)，实现vacc runstream推理，结果绘图
```
python ../build_in/vsx/vsx_inference.py \
    --model_prefix_path ./deploy_weights/mask2former_run_stream_fp16/mod \
    --image_dir /path/to/coco/det_coco_val \
    --vdsp_params_info ../build_in/vdsp_params/mask2former-vdsp_params.json \
    --save_dir runstream_output
```
- 注意替换命令行中相关参数为实际路径


### step.5 精度性能测试
1. 精度测试
- 基于Mask2Former仓库实现精度评估，修改库内代码，使用vacc模型替换原始模型，批量推理可获得mAP精度信息

    - 首先，撤回之前为了导出onnx模型，对Mask2Former和Detectron仓库做的修改
        ```bash
        cd /path/to/Mask2Former
        git apply -R mask2former_git_export_onnx.patch

        cd /path/to/Detectron2
        git apply -R detectron_git_export_onnx.patch
        ```

    - 然后，应用上跑数据集时需要的改动，主要涉及输入尺寸固定；限定Mask2Former使用cpu模式；vacc模型加载；vacc推理
        - [mask2former_git_run_dataset.patch](./official/mask2former_git_run_dataset.patch)
        - [detectron_git_run_dataset.patch](./official/detectron_git_run_dataset.patch)；

        ```bash
        cd /path/to/Mask2Former
        git apply mask2former_git_run_dataset.patch
        
        cd /path/to/Detectron
        git apply detectron_git_run_dataset.patch
        ```
        
    - 手动修改`Mask2Former/mask2former/maskformer_model.py`脚本中开头的三件套路径
    - 执行批量精度测试脚本

        ```bash
        cd /path/to/Mask2former/

        # # Mask2Former测试数据集路径是通过环境变量映射的，指定到coco文件夹的上一层
        # # coco/
             # annotations/
                # instances_val2017.json
             # val2017/
        export DETECTRON2_DATASETS=./datasets/coco/../

        python train_net.py \
        --config-file path/to/Mask2former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
        --eval-only \
        MODEL.WEIGHTS /path/to/weight/model_final_3c8ec9.pkl
        ```

        ```
        # Evaluation results for segm

        # torch soure shape
        |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
        |:------:|:------:|:------:|:------:|:------:|:------:|
        | 43.674 | 66.046 | 46.929 | 23.440 | 47.155 | 64.756 |

        # torch fixed shape in [1024, 1024]
        |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
        |:------:|:------:|:------:|:------:|:------:|:------:|
        | 42.220 | 63.876 | 45.487 | 22.243 | 45.556 | 63.283 |

        # vacc runmodel fixed shape in [1024, 1024]
        |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
        |:------:|:------:|:------:|:------:|:------:|:------:|
        | 42.183 | 63.956 | 45.446 | 22.230 | 45.526 | 63.272 |
        ```

2. 性能测试
- vamp测试
    ```bash
    # VA1L Devices

    同步模式可获得更准确的最小时延数据：
    vamp -m vacc_deploy/mask2former-fp16-none-1_3_1024_1024-vacc/mod --vdsp_params ../build_in/vdsp_params/mask2former-vdsp_params.json -p 1 -b 1 -i 1 -d 6 --forward_mode 1 --iterations 64
    ```



