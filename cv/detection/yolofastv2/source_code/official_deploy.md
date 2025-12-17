### step.1 获取预训练模型

官方提供[onnx转换脚本](https://github.com/dog-qiuqiu/Yolo-FastestV2/blob/main/pytorch2onnx.py)，直接运行`python pytorch2onnx.py`即可，运行前，需要修改`model/detector.py`中的[代码](https://github.com/dog-qiuqiu/Yolo-FastestV2/blob/main/model/detector.py)，如下

```python
if self.export_onnx:
   out_reg_2 = out_reg_2#.sigmoid()
   out_obj_2 = out_obj_2#.sigmoid()
   # out_cls_2 = F.softmax(out_cls_2, dim = 1)

   out_reg_3 = out_reg_3#.sigmoid()
   out_obj_3 = out_obj_3#.sigmoid()
   # out_cls_3 = F.softmax(out_cls_3, dim = 1)

   print("export onnx ...")
   return out_reg_2, out_obj_2, out_cls_2, out_reg_3, out_obj_3, out_cls_3
   '''return torch.cat((out_reg_2, out_obj_2, out_cls_2), 1).permute(0, 2, 3, 1), \
            torch.cat((out_reg_3, out_obj_3, out_cls_3), 1).permute(0, 2, 3, 1)  '''

```

同时，为了生成`torchscript`模型文件，可以在`pytorch2onnx.py`脚本中加入以下代码

```python
input_shape = (1, 3, cfg["height"], cfg["width"])
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

# scripted_model = torch.jit.script(net)
torch.jit.save(scripted_model, 'model.torchscript.pt')
```

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [official_yolofastv2.yaml](../build_in/build/official_yolofastv2.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd yolofastv2
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_yolofastv2.yaml
    ```

### step.4 模型推理

- 参考[yolofastv2_vsx.py](../build_in/vsx/python/yolofastv2_vsx.py)生成预测的txt结果

    ```
    python ../build_in/vsx/python/yolofastv2_vsx.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/official_yolofastv2_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-yolofastv2-vdsp_params.json \
        --label_txt path/to/coco.txt \
        --save_dir ./infer_output \
        --device 0
    ```

- 参考[eval_map.py](../../common/eval/eval_map.py)，精度统计
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt infer_output
    ```

    <details><summary>点击查看精度统计结果</summary>

    ```
    # fp16
    DONE (t=0.88s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.097
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.174
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.096
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.006
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.068
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.189
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.099
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.123
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.008
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.084
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.246
    {'bbox_mAP': 0.097, 'bbox_mAP_50': 0.174, 'bbox_mAP_75': 0.096, 'bbox_mAP_s': 0.006, 'bbox_mAP_m': 0.068, 'bbox_mAP_l': 0.189, 'bbox_mAP_copypaste': '0.097 0.174 0.096 0.006 0.068 0.189'}

    # int8
    DONE (t=0.86s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.079
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.139
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.080
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.003
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.047
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.154
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.080
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.098
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.098
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.004
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.056
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.199
    {'bbox_mAP': 0.079, 'bbox_mAP_50': 0.139, 'bbox_mAP_75': 0.08, 'bbox_mAP_s': 0.003, 'bbox_mAP_m': 0.047, 'bbox_mAP_l': 0.154, 'bbox_mAP_copypaste': '0.079 0.139 0.080 0.003 0.047 0.154'}
    ```

    </details>

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-yolofastv2-vdsp_params.json](../build_in/vdsp_params/official-yolofastv2-vdsp_params.json)：
    ```bash
    vamp -m deploy_weights/official_yolofastv2_int8/mod --vdsp_params ../build_in/vdsp_params/official-yolofastv2-vdsp_params.json -i 2 p 2 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```

    - vamp推理，获得npz结果
    ```bash
    vamp -m deploy_weights/official_yolofastv2_int8/mod --vdsp_params ../build_in/vdsp_params/official-yolofastv2-vdsp_params.json -i 2 p 2 -b 1 --datalist npz_datalist.txt --path_output npz_output
    ```
    
    - 解析npz文件，基于[npz_decode.py](../build_in/npz_decode.py)
    ```bash
    python ../build_in/npz_decode.py --txt result_npz --label_txt path/to/coco.txt --input_image_dir path/to/coco_val2017 --model_size 352 352 --vamp_datalist_path npz_datalist.txt --vamp_output_dir npz_output
    ```

    - 统计精度，基于[eval_map.py](../../common/eval/eval_map.py)
    ```bash
        python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt result_npz
    ```