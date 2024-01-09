### step.1 获取预训练模型

1. vacc暂不支持mmyolo库算法后处理，因此模型转换时需进行截断。对于yolov5算法来说，可以通过修改`YOLOv5HeadModule`模块截断后处理，代码修改具体如下

    ```python
    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        assert len(x) == self.num_levels
        out = []
        for i in range(3):
            out.append(self.convs_pred[i](x[i]))
        return tuple(out)
        # return multi_apply(self.forward_single, x, self.convs_pred)

    def forward_single(self, x: Tensor,
                        convs: nn.Module) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""

        pred_map = convs(x)
        '''bs, _, ny, nx = pred_map.shape
        pred_map = pred_map.view(bs, self.num_base_priors, self.num_out_attrib,
                                    ny, nx)

        cls_score = pred_map[:, :, 5:, ...].reshape(bs, -1, ny, nx)
        bbox_pred = pred_map[:, :, :4, ...].reshape(bs, -1, ny, nx)
        objectness = pred_map[:, :, 4:5, ...].reshape(bs, -1, ny, nx)

        return cls_score, bbox_pred, objectness'''
        return pred_map

    ```

2. mmyolo模型转换可以通过mmdeploy工具转换，也可通过mmyolo调用模型进行转换。这里给出通过修改mmyolo中`image_demo.py`脚本调用模型进行转换的脚本，如下

    ```python
    import os
    import mmcv
    import torch
    import thop
    from thop import clever_format
    from mmdet.apis import inference_detector, init_detector

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print(model)

    input_shape = (1, 3, 1280, 1280)
    input_data = torch.randn(input_shape).cuda()
    
    flops, params = thop.profile(model, inputs=(input_data,))
    print("flops(G):", "%.3f" % (flops / 900000000 * 2))
    flops,params = clever_format([ flops / 900000000 * 2,params], "%.3f")
    print("params:", params)
    
    model_name = 'yolov5l6-1280'
    import torch
    dynamic_axes = {
        'input': {0: '-1'},  # 这么写表示NCHW都会变化
    }

    input_shape = (1, 3, 1280, 1280)
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model.cpu(), input_data).eval()

    # scripted_model = torch.jit.script(net)
    torch.jit.save(scripted_model, model_name + '.torchscript.pt')

    input_names = ["input"]
    output_names = ["output"]
    inputs = torch.randn(1, 3, 1280, 1280)

    torch_out = torch.onnx._export(model.cpu(), inputs, model_name+'.onnx', export_params=True, verbose=False,
                                input_names=input_names, output_names=output_names, opset_version=10, dynamic_axes=dynamic_axes)

    ```

3. 通过运行如下脚本即可生成onnx与torchscript模型文件

    ```bash
    python demo/image_demo.py ~/project/yolov5/data/images/bus.jpg configs/yolov5/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco.py models/yolov5/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_234308-7a2ba6bf.pth
    ```

### step.2 准备数据集
- 准备[COCO](https://cocodataset.org/#download)数据集


### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[config_mmyolo.yaml](../vacc_code/build/config_mmyolo.yaml)：
    ```bash
    vamc build ../vacc_code/build/config_mmyolo.yaml
    ```

### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    ```bash
    vamp -m deploy_weights/yolo5s-int8-kl_divergence-3_640_640-vacc/yolo5s --vdsp_params ../vacc_code/vdsp_params/mmyolo-yolo5s-vdsp_params.json -i 2 p 2 -b 1
    ```
4. npz结果输出
    ```bash
    vamp -m deploy_weights/yolo5s-int8-kl_divergence-3_640_640-vacc/yolo5s --vdsp_params ../vacc_code/vdsp_params/mmyolo-yolo5s-vdsp_params.json -i 2 p 2 -b 1 --datalist datasets/coco_npz_datalist.txt --path_output npz_output
    ```
5. [vamp_decode_mmyolo.py](../vacc_code/vdsp_params/vamp_decode_mmyolo.py)，解析vamp输出的npz文件，进行绘图和保存txt结果
    ```bash
    python ../vacc_code/vdsp_params/vamp_decode_mmyolo.py --txt result_npz --label_txt datasets/coco.txt --input_image_dir datasets/coco_val2017 --model_size 640 640 --vamp_datalist_path datasets/coco_npz_datalist.txt --vamp_output_dir npz_output
    ```
6. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt path/to/vamp_draw_output
   ```
