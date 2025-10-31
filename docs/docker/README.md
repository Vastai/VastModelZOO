# Docker Quick Start

## 获取镜像

```bash
docker pull harbor.vastaitech.com/ai_deliver/vastmodelzoo-base:202510
```

若算法模型（如LLM、VLM方向模型）基于`VastGenX`部署，则需要拉取`vastmodelzoo-vastgenx`

```bash
docker pull harbor.vastaitech.com/ai_deliver/vastmodelzoo-vastgenx:202510
```

## 启动镜像

```bash
docker run -it --ipc=host --privileged --name=vastmodelzoo harbor.vastaitech.com/ai_deliver/vastmodelzoo-base:202510 bash
```

> docker内部署有vamc、deploy两个conda环境，可分别用于模型转换、精度/性能测试

`vastmodelzoo-vastgenx`镜像启动命令参考如下：
```bash
docker run --ipc=host -it --ipc=host --privileged --name=vastmodelzoo_vastgenx harbor.vastaitech.com/ai_deliver/vastmodelzoo-vastgenx:202510  bash
```

## 测试
> 以`resnet50`、`yolov8`、`Qwen2.5-7B`为例，其余模型可参考文档在容器内进行测试

### resnet50测试

#### Step.1 模型转换

```bash
conda activate vamc
vamc compile ../cv/classification/resnet/build_in/build/torch_resnet.yaml
```

#### Step.2 精度测试

```bash
conda activate deploy

# 推理得到结果
python ../cv/classification/common/vsx/classification.py \
    --infer_mode sync \
    --file_path path/to/ILSVRC2012_img_val \
    --model_prefix_path deploy_weights/torch_resnet50_run_model_fp16/mod \
    --vdsp_params_info ../cv/classification/resnet/build_in/vdsp_params/torchvision-resnet50-vdsp_params.json \
    --label_txt path/to/imagenet.txt \
    --save_dir ./runstream_output \
    --save_result_txt result.txt \
    --device 0

# 精度评估
python cv/classification/common/eval/eval_topk.py runstream_output/result.txt
```

#### Step.3 性能测试

```bash
vamp -m deploy_weights/torch_resnet50_run_stream_fp16/mod --vdsp_params ../cv/classification/resnet/build_in/vdsp_params/torchvision-resnet50-vdsp_params.json -i 8 -p 1 -b 2 -s [3,224,224]
```

### yolov8测试

#### Step.1 模型转换

```bash
conda activate vamc
vamc compile ../cv/detection/yolov8/build_in/build/ultralytics_yolov8.yaml
```

#### Step.2 精度测试

```bash
conda activate deploy

# 推理得到结果
python ../cv/detection/common/vsx/detection.py \
    --file_path /path/to/det_coco_val \
    --model_prefix_path deploy_weights/ultralytics_yolov8s_run_stream_fp16/mod \
    --vdsp_params_info ../cv/detection/yolov8/build_in/vdsp_params/ultralytics-yolov8s-vdsp_params.json \
    --label_txt /path/to/coco.txt \
    --save_dir ./runstream_output \
    --device 0

# 精度评估
python ../cv/detection/common/eval/eval_map.py --gt path/to/instances_val2017.json --txt ./runstream_output
```

#### Step.3 性能测试

```bash
vamp -m deploy_weights/ultralytics_yolov8s_run_stream_fp16/mod --vdsp_params ../cv/detection/yolov8/build_in/vdsp_params/ultralytics-yolov8s-vdsp_params.json -i 1 p 1 -b 1 -d 0
```

### Qwen2.5-7B测试

#### Step.0 模型下载及修改

从[huggingface](https://huggingface.co/Qwen/Qwen2.5-7B)或[modelscope](https://modelscope.cn/models/qwen/Qwen2.5-7B)下载模型，下载成功后基于[文档说明](../../llm/qwen2/README.md)添加所需文件，包括[modeling_qwen2_vacc.py](../../llm/qwen2/build_in/source_code/modeling_qwen2_vacc.py)、[configuration_qwen2_vacc.py](../../llm/qwen2/build_in/source_code/configuration_qwen2_vacc.py)、[quantization_vacc.py](../../llm/qwen2/build_in/source_code/quantization_vacc.py)，并参考[config_vacc.json](../../llm/qwen2/build_in/source_code/config_vacc.json)添加`config_vacc.json`，**需根据模型配置对齐**


#### Step.1 模型转换

在`vastmodelzoo-base`镜像中进行模型转换

```bash
conda activate vamc
vamc compile ../../llm/qwen2/build_in/build/hf_qwen2_fp16.yaml
```

#### Step.2 启动模型服务

参考[vastgenx说明文档](../vastgenx/README.md)在`harbor.vastaitech.com/ai_deliver/vastmodelzoo-vastgenx:202510`镜像中启动模型服务，命令如下：

```bash
vastgenx serve --model vacc_deploy/Qwen2-7B-fp16-tp4-1024-2048/ --port 9900 --llm_devices "[12,13,14,15]"
```

#### Step.3 精度测试及性能测试

参考[vastgenx说明文档](../vastgenx/README.md)进行精度测试及性能测试