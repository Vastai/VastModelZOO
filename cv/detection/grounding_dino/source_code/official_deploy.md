## Deploy

### step.1 获取模型

- 模型来源

    ```bash
    # 原始仓库
    - gitlab：https://github.com/IDEA-Research/GroundingDINO
    - commit: 856dde20aee659246248e20734ef9ba5214f5e44

    - config:
        - https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinB_cfg.py
        - https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
    - weight: 
        - https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
        - https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
    ```

- 环境配置

    ```shell
    # torch版本不一致，可能导致onnx存在差异

    conda create -n g_dino python==3.8
    conda activate g_dino

    # 安装依赖
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -r detection/grounding_dino/source_code/official/requirements.txt
    ```

- 两个子模型流程一致，以下基于`groundingdino_swint_ogc`描述导出和编译流程
- 原始模型需要拆分为三个子模型，text_encoder，image_encoder，decoder；需要适当修改原始模型forward，将修改后[official/groundingdino/models/GroundingDINO](./official/groundingdino/models/GroundingDINO)的三个脚本替换原始仓库同路径文件，即[groundingdino/models/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/tree/main/groundingdino/models/GroundingDINO)
- 基于以上修改后仓库，将脚本[export_onnx.py](./official/export_onnx.py)移动至{GroundingDINO}仓库根目录，导出完整onnx和简化onnx
- 基于[cut_onnx.py](./official/cut_onnx.py)，切分为三个子模型onnx
- 替换自定义算子
    - 基于[convert_custom_op_img_encoder.py](./official/convert_custom_op_img_encoder.py)，修改image_encoder模型内算子为自定义算子
    - 基于[convert_custom_op_decoder.py](./official/convert_custom_op_decoder.py)，修改decoder模型deform_attn、sine_position_embed等算子为自定义算子

- 后续完整推理时，embed部分依赖`bert-base-uncased`：https://huggingface.co/google-bert/bert-base-uncased
    - 如有网络，`export_onnx.py`步骤将自动下载至`~/.cache/huggingface/hub/models--bert-base-uncased`目录
    - 如无网络，可从huggingface下载仓库后，手动配置源仓库的对应config内的`text_encoder_type = "bert-base-uncased"`，值重写为本地下载后文件夹路径

### step.2 获取数据集
- 依据原始仓库，使用[coco val2017](https://cocodataset.org/#download)验证模型精度
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

### step.3 模型转换
1. 环境准备
    ```
    # 需要配置odsp环境变量
    cd /path/to/odsp_plugin/vastai/
    sudo ./build.sh

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/odsp_plugin/vastai/lib:/path/to/odsp_plugin/protobuf/lib/x86_64
    ```

2. 根据具体模型修改配置文件:
    - [image_encoder_build.yaml](../build_in/build/image_encoder_build.yaml)
    - [text_encoder_build.yaml](../build_in/build/text_encoder_build.yaml)
    - [decoder_build.yaml](../build_in/build/decoder_build.yaml)
    > - 注意当前模型，build_in只支持fp16精度

    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

3. 模型编译
    ```bash
    cd grounding_dino
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/image_encoder_build.yaml
    vamc compile ../build_in/build/text_encoder_build.yaml
    vamc compile ../build_in/build/decoder_build.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考： [grounding_dino_vsx.py](../build_in/vsx/python/grounding_dino_vsx.py)

    ```
    mkdir -p ./grounding_dino_out
    python3 ../build_in/vsx/python/grounding_dino_vsx.py \
        --txtmod_prefix deploy_weights/text_encoder_build_run_stream_fp16/mod \
        --txtmod_vdsp_params  ../build_in/vdsp_params/text_encoder-vdsp_params.json \
        --imgmod_prefix deploy_weights/image_encoder_build_run_stream_fp16/mod \
        --imgmod_vdsp_params ../build_in/vdsp_params/image_encoder-vdsp_params.json \
        --decmod_prefix deploy_weights/decoder_build_run_stream_fp16/mod \
        --tokenizer_path ./bert-base-uncased \
        --label_file /path/to/coco/coco.txt \
        --device_id  0 \
        --threshold 0.01 \
        --dataset_filelist ./det_coco_val.txt \
        --dataset_root /path/to/det_coco_val/ \
        --dataset_output_folder ./grounding_dino_out
    ```
    - det_coco_val.txt生成方法
    ```
    ls /path/to/det_coco_val | grep jpg > det_coco_val.txt
    ```
    
    - 参考：
    ```
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt ./grounding_dino_out
    ```

    ```
    # fp16
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.452
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.603
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.495
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.483
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.584
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.376
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.660
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.730
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.578
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.769
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.880
    {'bbox_mAP': 0.452, 'bbox_mAP_50': 0.603, 'bbox_mAP_75': 0.495, 'bbox_mAP_s': 0.316, 'bbox_mAP_m': 0.483, 'bbox_mAP_l': 0.584, 'bbox_mAP_copypaste': '0.452 0.603 0.495 0.316 0.483 0.584'}
    ```

### step.5 性能测试
1. 参考：[grounding_dino_text_enc_prof.py](../build_in/vsx/python/grounding_dino_text_enc_prof.py)测试text_encoder性能
    ```
    python3 ../build_in/vsx/python/grounding_dino_text_enc_prof.py \
        -m deploy_weights/text_encoder_build_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/text_encoder-vdsp_params.json \
        --tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/bert-base-uncased \
        --device_ids [0] \
        --batch_size 1 \
        --instance 1 \
        --iterations 1000 \
        --percentiles [50,90,95,99] \
        --input_host 1 \
        --queue_size 1 
    ```

2. 参考：[grounding_dino_image_enc_prof.py](../build_in/vsx/python/grounding_dino_image_enc_prof.py)测试image_encoder性能
    ```
    python3 ../build_in/vsx/python/grounding_dino_image_enc_prof.py \
        -m deploy_weights/image_encoder_build_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/image_encoder-vdsp_params.json \
        --device_ids [0] \
        --batch_size 1 \
        --instance 1 \
        --iterations 10 \
        --percentiles [50,90,95,99] \
        --input_host 1 \
        --queue_size 1 
    ```