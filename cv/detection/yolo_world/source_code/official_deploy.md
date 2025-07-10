## Deploy

### step.1 获取模型

- 模型来源

    ```bash
    # 原始仓库
    - gitlab：https://github.com/AILab-CVC/YOLO-World
    - commit: a9d9ef520729798b475c39a147b0913e3fcb5795

    # 因vacc在text_backbone部分不支持batch推理，所以需要对源码进行修改；
    # 修改内容可查看Chao Shi的最新几个提交，主要涉及多个模块的forward部分text输出处理，需要拆分，每次输入一个
    - gitee: https://gitee.com/tutu96177/YOLO-World/tree/export_model_yolov8_and_clip_model/
    - commit: fef90a96a9bd19727d13d1cdce3ec6dda8837b35

    # 相比原始仓库config，修改neck和head部分，增加参数use_einsum=False，不使用torch.einsum算子，编译后续编译
    - config: https://gitee.com/tutu96177/YOLO-World/blob/export_model_yolov8_and_clip_model/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py
    - weight: https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth
    ```

- 环境配置

    ```shell
    # torch版本不一致，可能导致onnx存在差异

    conda create -n yolo python==3.8
    conda activate yolo

    # 安装cpu版本torch
    pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0

    # 安装其它依赖
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip3 install transformers tokenizers numpy opencv-python onnx onnxsim onnxruntime chardet decorator
    pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.11/index.html

    pip3 install openmim supervision==0.18.0 mmdet>=3.0.0 mmengine>=0.7.1 mmyolo==0.6.0
    pip3 install lvis  # 如后续有报错，参考https://github.com/lvis-dataset/lvis-api/pull/38

    # 安装YOLO-World库
    git clone https://gitee.com/tutu96177/YOLO-World.git
    git checkout fef90a96a9bd19727d13d1cdce3ec6dda8837b35
    cd path/to/YOLO-World
    pip3 install -e .
    ```

- 基于以上修改后仓库，将脚本[onnx_export.py](./official/onnx_export.py)移动至{YOLO-World/deploy}目录，分两次导出两个onnx
    - 第一次，导出完整onnx（后续从里面截取image_backbone部分）
        ```shell
        python multimodal/yolo_world/source_code/official/onnx_export.py \
            --config configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py \
            --checkpoint yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth \
            --export image \
            --opset 11 --model-only --work-dir ./output_onnx
        ```
    - 第二次，导出text部分bs1的onnx（后续从里面截取text_backbone部分）
        ```shell
        python multimodal/yolo_world/source_code/official/onnx_export.py \
            --config configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py \
            --checkpoint yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth \
            --export text \
            --opset 11 --model-only --work-dir ./output_onnx
        ```
    - 在`./output_onnx`目录下生成两个onnx，分别以`_image.onnx`和`_text.onnx`结尾，对应上面两次导出的onnx

    > 注：如遇到导出模型错误，需要把site-packages/mmdet/models/detectors/base.py中的文件内容，替换为[base.py](./official/base.py)

- 对以上导出的两个onnx进行简化
    ```shell
    onnxsim ./output_onnx/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6_image.onnx \
            ./output_onnx/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6_image-sim.onnx \
            --overwrite-input-shape images:1,3,1280,1280 input_ids:1203,16 attention_mask:1203,16

    onnxsim ./output_onnx/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6_text.onnx \
            ./output_onnx/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6_text-sim.onnx \
            --overwrite-input-shape images:1,3,1280,1280 input_ids:1,16 attention_mask:1,16
    ```

- 对以上两个简化onnx进行截断，独立出image_backbone和text_backbone，方便后续vacc编译；增加后缀`_sub`在同目录下生成onnx
    ```shell
    python multimodal/yolo_world/source_code/official/onnx_sub.py \
        --onnx_file ./output_onnx/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6_image-sim.onnx \
        --export image

    python multimodal/yolo_world/source_code/official/onnx_sub.py \
        --onnx_file ./output_onnx/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6_text-sim.onnx \
        --export text
    ```

### step.2 获取数据集
- 依据原始仓库，使用[coco val2017](https://cocodataset.org/#download)验证模型精度
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [lvis_v1_class_texts.json](https://github.com/AILab-CVC/YOLO-World/blob/master/data/texts/lvis_v1_class_texts.json)
- [lvis_v1_minival_inserted_image_name.json](https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json)
- [tokenizer](https://hf-mirror.com/openai/clip-vit-base-patch32/tree/main)

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [image_build.yaml](../build_in/build/image_build.yaml)
    - [text_build.yaml](../build_in/build/text_build.yaml)

    > - runmodel推理，编译参数`backend.type: tvm_runmodel`
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 模型编译
    ```bash
    cd yolo_world
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/image_build.yaml
    vamc compile ../build_in/build/text_build.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考： [yolo_world_vsx.py](../build_in/vsx/python/yolo_world_vsx.py)

    ```
    python3 ../build_in/vsx/python/yolo_world_vsx.py \
        --imgmod_prefix  deploy_weights/image_build_run_stream_fp16/mod \
        --imgmod_vdsp_params ../build_in/vdsp_params/image-vdsp_params.json \
        --txtmod_prefix deploy_weights/text_build_run_stream_fp16/mod \
        --txtmod_vdsp_params  ../build_in/vdsp_params/text-vdsp_params.json \
        --tokenizer_path ./clip-vit-base-patch32 \
        --device_id  0 \
        --max_per_image 300 \
        --score_thres  0.001 \
        --iou_thres  0.7 \
        --nms_pre  30000 \
        --label_file ./lvis_v1_class_texts.json \
        --dataset_root /path/to/det_coco_val \
        --dataset_filelist ./det_coco_val.txt \
        --dataset_output_file yoloworld_dataset_result.json
    ```

    - det_coco_val.txt生成方法
    ```
    ls /path/to/det_coco_val | grep jpg > det_coco_val.txt
    ```

    - 参考：[eval_map.py](../../common/eval/eval_map.py)，进行精度统计
    ```bash
    python3 ../source_code/eval_lvis.py  \
        --path_res yoloworld_dataset_result.json \
        --path_ann_file /path/to/lvis_v1_minival_inserted_image_name.json
    ```
    测试参考精度如下：
    ```
    # fp16
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=all] = 0.348
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=300 catIds=all] = 0.457
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=300 catIds=all] = 0.379
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=     s | maxDets=300 catIds=all] = 0.255
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=     m | maxDets=300 catIds=all] = 0.456
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=     l | maxDets=300 catIds=all] = 0.540
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=  r] = 0.292
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=  c] = 0.331
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=  f] = 0.372
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=all] = 0.452
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=     s | maxDets=300 catIds=all] = 0.293
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=     m | maxDets=300 catIds=all] = 0.566
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=     l | maxDets=300 catIds=all] = 0.673~
    ```

### step.5 性能测试
1. 参考[yolo_world_image_prof.py](../build_in/vsx/python/yolo_world_image_prof.py)，测试image模型性能：
    ```
    python3 ../build_in/vsx/python/yolo_world_image_prof.py \
        --model_prefix deploy_weights/image_build_run_stream_fp16/mod  \
        --vdsp_params ../build_in/vdsp_params/image-vdsp_params.json \
        --device_ids  [0] \
        --batch_size  1 \
        --instance 1 \
        --iterations 20 \
        --percentiles "[50,90,95,99]" \
        --input_host 1 \
        --queue_size 1
    ```

2. 参考[yolo_world_text_prof.py](../build_in/vsx/python/yolo_world_text_prof.py)，测试text模型性能：
    ```
    python3 ../build_in/vsx/python/yolo_world_text_prof.py \
        --model_prefix deploy_weights/text_build_run_stream_fp16/mod \
        --vdsp_params  ../build_in/vdsp_params/text-vdsp_params.json \
        --tokenizer_path ./clip-vit-base-patch32 \
        --device_ids  [0] \
        --batch_size  1 \
        --instance 1 \
        --iterations 2000 \
        --percentiles "[50,90,95,99]" \
        --input_host 1 \
        --queue_size 1
    ```

### appending
- `text`部分先经AutoTokenizer模型`openai/clip-vit-base-patch32`编码后，再`get_bert_preprocess`后，进入`text_backbone`inference；
    - 此模型在代码内自动下载，设置hf镜像，可加速下载：`export HF_ENDPOINT=https://hf-mirror.com`
    - 也可手动下载权重`https://hf-mirror.com/openai/clip-vit-base-patch32/tree/main`，替换路径[runmodel.py#L111](../build_in/runmodel/runmodel.py#L111)
- 部分脚本涉及onnx模型输入输出节点的名称设置，和torch版本有关系




