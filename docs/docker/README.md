# Docker Quick Start

## 获取镜像

```bash
docker pull harbor.vastaitech.com/ai_deliver/vastmodelzoo-base:202507
```

## 启动镜像

```bash
docker run -it --ipc=host --privileged --name=vastmodelzoo harbor.vastaitech.com/ai_deliver/vastmodelzoo-base:202507 bash
```

> docker内部署有vamc、deploy两个conda环境，可分别用于模型转换、精度/性能测试

## 测试
> 以`resnet50`、`yolov8`为例，其余模型可参考文档在容器内进行测试

### resnet50测试

### Step.1 模型转换

```bash
conda activate vamc
vamc compile ../cv/classification/resnet/build_in/build/torch_resnet.yaml
```

### Step.2 精度测试

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

### Step.3 性能测试

```bash
vamp -m deploy_weights/torch_resnet50_run_stream_fp16/mod --vdsp_params ../cv/classification/resnet/build_in/vdsp_params/torchvision-resnet50-vdsp_params.json -i 8 -p 1 -b 2 -s [3,224,224]
```

### yolov8测试

### Step.1 模型转换

```bash
conda activate vamc
vamc compile ../cv/detection/yolov8/build_in/build/ultralytics_yolov8.yaml
```

### Step.2 精度测试

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

### Step.3 性能测试

```bash
vamp -m deploy_weights/ultralytics_yolov8s_run_stream_fp16/mod --vdsp_params ../cv/detection/yolov8/build_in/vdsp_params/ultralytics-yolov8s-vdsp_params.json -i 1 p 1 -b 1 -d 0