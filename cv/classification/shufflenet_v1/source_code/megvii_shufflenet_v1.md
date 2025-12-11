### step.1 获取模型
megvii官方实现，可以参考以下代码实现生成onnx或torchscript中间表示
```python
import torch
from network import ShuffleNetV1

def get_model(model_size, group):
    model = ShuffleNetV1(group=group, model_size=model_size)
    if group == 3:
        checkpoint = torch.load(f'models/Group{group}/models/{model_size}.pth.tar', map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(f'models/Group{group}/models/snetv1_group{group}_{model_size}.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
    model.eval()
    return model

if __name__ == '__main__':
    model_size = ['0.5x', '1.0x', '1.5x', '2.0x']
    group = [3, 8]
    for g in group:
        for m in model_size:
            model_save = 'export_model/Group' + str(g) + '_' + str(m)
            model = get_model(m, g)

            input_shape = (1, 3, 224, 224)
            input_data = torch.randn(input_shape)
            model(input_data)
            scripted_model = torch.jit.trace(model, input_data)

            # scripted_model = torch.jit.script(net)
            torch.jit.save(scripted_model, model_save + '.torchscript.pt')

            input_names = ["input"]
            output_names = ["output"]
            inputs = torch.randn(1, 3, 224, 224)

            torch_out = torch.onnx._export(model, inputs, model_save + '.onnx', export_params=True, verbose=False, opset_version=10,
                                        input_names=input_names, output_names=output_names)
```


### step.2 获取数据集
- [校准数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../common/label/imagenet.txt)
- [label_dict](../../common/label/imagenet1000_clsid_to_human.txt)

### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [megvii_shufflenet_v1.yaml](../build_in/build/megvii_shufflenet_v1.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd shufflenet_v1
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/megvii_shufflenet_v1.yaml
    ```

### step.4 模型推理

1. runstream
    - 参考：[classification.py](../../common/vsx/python/classification.py)
    ```bash
    python ../../common/vsx/python/classification.py \
        --file_path path/to/ILSVRC2012_img_val \
        --model_prefix_path deploy_weights/megvii_shufflenetv1_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/megvii-Group3_1.0x-vdsp_params.json \
        --label_txt path/to/imagenet.txt \
        --save_dir ./runstream_output \
        --save_result_txt result.txt \
        --device 0
    ```

    - 精度评估
    ```
    python ../../common/eval/eval_topk.py ./runstream_output/result.txt
    ```

    ```
    # fp16
    top1_rate: 66.746 top5_rate: 87.198

    # int8
    top1_rate: 65.564 top5_rate: 86.314
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置[megvii-Group3_1.0x-vdsp_params.json](../build_in/vdsp_params/megvii-Group3_1.0x-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/megvii_shufflenetv1_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/megvii-Group3_1.0x-vdsp_params.json  -i 8 -p 1 -b 2 -s [3,224,224]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致
    
    - 数据准备，生成推理数据`npz`以及对应的`dataset.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path ILSVRC2012_img_val --target_path  input_npz  --text_path imagenet_npz.txt
    ```

    - vamp推理获取npz文件
    ```
    vamp -m deploy_weights/megvii_shufflenetv1_run_stream_fp16/mod --vdsp_params ../build_in/vdsp_params/megvii-Group3_1.0x-vdsp_params.json  -i 8 -p 1 -b 22 -s [3,224,224] --datalist imagenet_npz.txt --path_output output
    ```

    - 解析输出结果用于精度评估，参考：[vamp_npz_decode.py](../../common/eval/vamp_npz_decode.py)
    ```bash
    python  ../../common/eval/vamp_npz_decode.py imagenet_npz.txt output imagenet_result.txt imagenet.txt
    ```
    
    - 精度评估，参考：[eval_topk.py](../../common/eval/eval_topk.py)
    ```bash
    python ../../common/eval/eval_topk.py imagenet_result.txt
    ```
