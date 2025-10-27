# pytorch

### step.1 获取预训练模型

1. 修改原始库中`retinanet/model.py`，替换为该目录下[pytorch/model.py](./pytorch/model.py)文件

2. 运行如下脚本进行模型转换，生成onnx格式

    ```python
    import torch
    from retinanet.model import resnet50

    model_path = 'weights/coco_resnet_50_map_0_335_state_dict.pt'
    model = resnet50(num_classes=80,)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.training = False
    model.eval()

    input_shape = (1, 3, 1024, 1024)
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data)
    torch.jit.save(scripted_model, 'weights/retinanet.torchscript.pt')

    input_names = ["input"]

    torch_out = torch.onnx._export(model, input_data, 'weights/retinanet.onnx', export_params=True, verbose=False,
                                input_names=input_names, opset_version=11)
    ```

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [pytorch_retinanet.yaml](../build_in/build/pytorch_retinanet.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd retinanet
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/pytorch_retinanet.yaml
    ```

### step.4 模型推理

1. runstream

    - 参考[pytorch_retinanet_vsx.py](../build_in/vsx/python/pytorch_retinanet_vsx.py)生成预测的txt结果

    ```
    python ../build_in/vsx/python/pytorch_retinanet_vsx.py \
        --file_path path/to/coco_val2017 \
        --model_prefix_path deploy_weights/pytorch_retinanet_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/pytorch-retinanet_resnet50-vdsp_params.json \
        --label_txt path/to/coco.txt \
        --save_dir ./runstream_output \
        --device 0
    ```

    - 参考[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt runstream_output
    ```

    <details><summary>点击查看精度测试结果</summary>

    ```
    # fp16
    DONE (t=9.84s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.341
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.503
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.370
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.182
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.379
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.451
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.297
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.479
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.533
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.587
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.673
    {'bbox_mAP': 0.341, 'bbox_mAP_50': 0.503, 'bbox_mAP_75': 0.37, 'bbox_mAP_s': 0.182, 'bbox_mAP_m': 0.379, 'bbox_mAP_l': 0.451, 'bbox_mAP_copypaste': '0.341 0.503 0.370 0.182 0.379 0.451'}

    # int8
    DONE (t=9.69s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.338
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.498
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.367
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.180
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.376
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.446
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.296
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.478
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.532
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.584
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.671
    {'bbox_mAP': 0.338, 'bbox_mAP_50': 0.498, 'bbox_mAP_75': 0.367, 'bbox_mAP_s': 0.18, 'bbox_mAP_m': 0.376, 'bbox_mAP_l': 0.446, 'bbox_mAP_copypaste': '0.338 0.498 0.367 0.180 0.376 0.446'}
    ```

    </details>

### step.5 性能精度测试

1. 性能测试
    ```bash
    vamp -m deploy_weights/pytorch_retinanet_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/pytorch-retinanet_resnet50-vdsp_params.json -i 1 -b 1 -d 0 -p 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path path/to/coco_val2017 \
        --target_path  path/to/coco_val2017_npz  \
        --text_path npz_datalist.txt
    ```
    
    - vamp推理，获取npz结果输出
    ```bash
    vamp -m deploy_weights/retinanet-int8-max-1_3_1024_1024-debug/retinanet \
        --vdsp_params ../build_in/vdsp_params/pytorch-retinanet_resnet50-vdsp_params.json  \
        -i 1 -b 1 -d 0 -p 1 \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 精度统计，参考：[eval_map.py](../../common/eval/eval_map.py)
    ```bash
    python ../../common/eval/eval_map.py \
        --gt path/to/instances_val2017.json \
        --txt npz_output
    ```
