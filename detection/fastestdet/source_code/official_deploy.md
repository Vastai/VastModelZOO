## official

```bash
git clone https://github.com/dog-qiuqiu/FastestDet.git
cd FastestDet
git checkout 50473cd155cb088aa4a99e64ff6a4b3c24fa07e1
```

### step.1 获取预训练模型

官方提供onnx/torchscript转换脚本，直接运行`test.py`脚本即可，运行前，需要修改`module/custom_layers.py`中的代码，如下

```python
class DetectHead(nn.Module):
   def __init__(self, input_channels, category_num):
      super(DetectHead, self).__init__()
      self.conv1x1 =  Conv1x1(input_channels, input_channels)

      self.obj_layers = Head(input_channels, 1)
      self.reg_layers = Head(input_channels, 4)
      self.cls_layers = Head(input_channels, category_num)

      #self.sigmoid = nn.Sigmoid()
      #self.softmax = nn.Softmax(dim=1)
      
   def forward(self, x):
      x = self.conv1x1(x)
      
      obj = self.obj_layers(x)
      reg = self.reg_layers(x)
      cls = self.cls_layers(x)
      return obj, reg, cls

      # return torch.cat((obj, reg, cls), dim =1)

```

### step.2 准备数据集
- 准备[COCO](https://cocodataset.org/#download)数据集


### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[config.yaml](../vacc_code/build/config.yaml)：
    ```bash
    vamc build ../vacc_code/build/config.yaml
    ```

### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    ```bash
    vamp -m deploy_weights/fastestdet-int8-kl_divergence-3_352_352-vacc/mod --vdsp_params ../vacc_code/vdsp_params/official-fastestdet-vdsp_params.json -i 2 p 2 -b 1
    ```
4. npz结果输出
    ```bash
    vamp -m deploy_weights/fastestdet-int8-kl_divergence-3_352_352-vacc/fastestdet --vdsp_params ../vacc_code/vdsp_params/official-fastestdet-vdsp_params.json -i 2 p 2 -b 1 --datalist datasets/coco_npz_datalist.txt --path_output npz_output
    ```
5. [vamp_decode.py](../vacc_code/vdsp_params/vamp_decode.py)，解析vamp输出的npz文件，进行绘图和保存txt结果
    ```bash
    python ../vacc_code/vdsp_params/vamp_decode.py --label_txt ../eval/coco.txt --input_image_dir ~/datasets/detection/coco_val2017 --model_size 352 352 --vamp_datalist_path ~/datasets/detection/datalist_npz.txt --vamp_output_dir npz_out
    ```
6. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt path/to/vamp_draw_output
   ```