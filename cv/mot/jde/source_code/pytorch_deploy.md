# pytorch

### step.1 获取预训练模型

基于[JDE](https://github.com/Zhongdao/Towards-Realtime-MOT/tree/master)，修改代码截断后处理

1. 修改`models.py`中`Darknet`类`Forward`函数，如下
    ```python
    def forward(self, x, targets=None, targets_len=None):
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        is_training = (targets is not None) and (not self.test_emb)
        #img_size = x.shape[-1]
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    if layer_i[0] == -3 and layer_i[1] == -1:
                        output.append(layer_outputs[-3])
                        output.append(layer_outputs[-1])
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                if is_training:  # get loss
                    targets = [targets[i][:int(l)] for i,l in enumerate(targets_len)]
                    x, *losses = module[0](x, self.img_size, targets, self.classifier)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                elif self.test_emb:
                    if targets is not None:
                        targets = [targets[i][:int(l)] for i,l in enumerate(targets_len)]
                    x = module[0](x, self.img_size, targets, self.classifier, self.test_emb)
                else:  # get detections
                    x = module[0](x, self.img_size)
                # output.append(x)
            layer_outputs.append(x)
        return output
    ```
2. 在`tracker/multitracker.py`中`JDETracker`类`update`函数中添加如下代码，导出onnx格式模型文件

    ```python
    input_names = ["input"]
    output_names = ["output"]

    torch_out = torch.onnx._export(self.model, im_blob, 'jde_864x480_uncertainty.onnx', export_params=True, verbose=False,
                                input_names=input_names, output_names=output_names, opset_version=11)
    ```


3. 运行如下脚本转换模型至onnx格式

    ```
    python track.py --cfg ./cfg/yolov3_864x480.cfg --weights model/jde_864x480_uncertainty.pt --test-mot16 --save-images
    ```


### step.2 准备数据集

- [校准数据集](https://motchallenge.net/data/MOT16/)
- [评估数据集](https://motchallenge.net/data/MOT16/)

### step.3 模型转换

1. 根据具体模型,编译配置
    - [pytorch_jde.yaml](../build_in/build/pytorch_jde.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd jde
    mkdir workspace
    cd workspace
    vamc build ../build_in/build/pytorch_jde.yaml
    ```

### step.4 模型推理

基于[jde_vsx.py](../build_in/vsx/python/jde_vsx.py)测试数据集精度，测试检测跟踪整体pipeline，脚本如下

```bash
python ../build_in/vsx/python/jde_vsx.py
```

### step.5 性能测试

**Note:** JDE多目标跟踪算法包括检测、track两个流程，modelzoo只提供算法模型的性能测试，整体pipeline性能可使用vastpipe测试。精度采用python脚本计算，如上`Step.4`

1. 性能测试
    ```bash
    vamp -m deploy_weights/jde_1088x608_uncertainty-int8-max-1_3_608_1088-vacc/jde_1088x608_uncertainty --vdsp_params vdsp_params.json -i 1 -b 1 -d 0 -p 1
    ```
