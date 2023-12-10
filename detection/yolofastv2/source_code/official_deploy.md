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
- 准备[COCO](https://cocodataset.org/#download)数据集


### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[config_official.yaml](../vacc_code/build/config_official.yaml)：
    ```bash
    vamc build ../vacc_code/build/config_official.yaml
    ```

### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    ```bash
    vamp -m deploy_weights/yolofastv2-int8-kl_divergence-3_352_352-vacc/yolofastv2 --vdsp_params ../vacc_code/vdsp_params/official-yolofastv2-vdsp_params.json -i 2 p 2 -b 1
    ```
4. npz结果输出
    ```bash
    vamp -m deploy_weights/yolofastv2-int8-kl_divergence-3_352_352-vacc/yolofastv2 --vdsp_params ../vacc_code/vdsp_params/official-yolofastv2-vdsp_params.json -i 2 p 2 -b 1 --datalist datasets/coco_npz_datalist.txt --path_output npz_output
    ```
5. [vamp_decode.py](../vacc_code/vdsp_params/vamp_decode.py)，解析vamp输出的npz文件，进行绘图和保存txt结果
    ```bash
    python ../vacc_code/vdsp_params/vamp_decode.py --txt result_npz --label_txt datasets/coco.txt --input_image_dir datasets/coco_val2017 --model_size 352 352 --vamp_datalist_path datasets/coco_npz_datalist.txt --vamp_output_dir npz_output
    ```
6. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt path/to/vamp_draw_output
   ```