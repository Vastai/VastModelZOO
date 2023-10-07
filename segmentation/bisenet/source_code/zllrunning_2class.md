
## zllrunning版本

### step.1 获取预训练模型
```
link: https://github.com/zllrunning/face-parsing.PyTorch
branch: master
commit: d2e684cf1588b46145635e8fe7bcc29544e5537e
```

- 训练模型，基于原始仓库，训练两个标签模型，即背景和cloth合并为标签0，人脸其它组分合并为1
- 按官方仓库准备数据集
- 修改数据加载部分，在[face_dataset.py#L55](https://github.com/zllrunning/face-parsing.PyTorch/blob/master/face_dataset.py#L55)的55行后，增加两行代码，将多标签进行合并
    ```python
    label[label == 16] = 0 # cloth
    label[label > 1] = 1  # other merge to one class
    ```
- 修改训练部分，[train.py#L56](https://github.com/zllrunning/face-parsing.PyTorch/blob/master/train.py#L56)，`n_classes=2，cropsize=[512, 512]`
- 开始训练，`CUDA_VISIBLE_DEVICES='2,3,' python -m torch.distributed.launch --nproc_per_node=2 train.py`
- 训练完成后，导出IR，一般在原始仓库内进行模型转为onnx或torchscript。在原仓库test或val脚本内，如[test.py](./face_parsing/test.py)，定义模型和加载训练权重后，添加以下脚本可实现：

    ```python
    checkpoint = save_pth

    input_shape = (1, 3, 512, 512)
    shape_dict = [("input", input_shape)]
    input_data = torch.randn(input_shape)

    scripted_model = torch.jit.trace(net, input_data).eval()
    scripted_model.save(checkpoint.replace(".pth", "-512.torchscript.pt"))
    scripted_model = torch.jit.load(checkpoint.replace(".pth", "-512.torchscript.pt"))

    # # onnx==10.0.0，opset 10
    # import onnx
    # torch.onnx.export(net, input_data, checkpoint.replace(".pth", "-512.onnx"), input_names=["input"], output_names=["output"], opset_version=11)
    # shape_dict = {"input": input_shape}
    # onnx_model = onnx.load(checkpoint.replace(".pth", "-512.onnx"))
    ```


### step.2 准备数据集
- 下载[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)数据集，解压
- 这个仓库的标签和数据集官方标签顺序不一样，需要用仓库脚本[prepropess_data.py](./face_parsing/prepropess_data.py)（参考自[源仓库](https://github.com/zllrunning/face-parsing.PyTorch/blob/master/prepropess_data.py)生成验证数据集）

### step.3 模型转换
1. 获取vamc模型转换工具
2. 根据具体模型修改配置文件，[zllrunning_config.yaml](../vacc_code/build/zllrunning_config.yaml)
3. 执行转换

   ```bash
    vamc build ../vacc_code/build/zllrunning_config.yaml
   ```

### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path CelebAMask-HQ/bisegnet_test_img \
    --target_path  CelebAMask-HQ/bisegnet_test_img_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[zllrunning-bisenet_2class-vdsp_params.json](../vacc_code/vdsp_params/zllrunning-bisenet_2class-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/bisenet_2class-int8-kl_divergence-3_512_512-vacc/bisenet_2class \
    --vdsp_params ../vacc_code/vdsp_params/zllrunning-bisenet_2class-vdsp_params.json \
    -i 2 p 2 -b 1
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/bisenet_2class-int8-kl_divergence-3_512_512-vacc/bisenet_2class \
    --vdsp_params vacc_code/vdsp_params/zllrunning-bisenet_2class-vdsp_params.json \
    -i 2 p 2 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [zllrunning_vamp_eval2.py](../vacc_code/vdsp_params/zllrunning_vamp_eval2.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/zllrunning_vamp_eval2.py \
    --src_dir CelebAMask-HQ/bisegnet_test_img \
    --gt_dir CelebAMask-HQ/bisegnet_test_mask \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 512 512 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


#### Tips

- 默认模型有三个输出，[model.py#L254](https://github.com/zllrunning/face-parsing.PyTorch/blob/master/model.py#L254)，导出IR时只返回`feat_out32`一个输出，可改善vaststream run速度（[pretrained_weights.json](../pretrained_weights.json)内提供的为单输出）
- 
    <details><summary>eval metrics</summary>

    ```
    torch 512 classes = 2
    ----------------- Total Performance --------------------
    Overall Acc:     0.9900743658153341
    Mean Acc :       0.9875138774962051
    FreqW Acc :      0.9803652727764909
    Mean IoU :       0.9767597274974249
    Overall F1:      0.9882216812155405
    ----------------- Class IoU Performance ----------------
    background      : 0.9676330876295614
    all_in_one_except_cloth : 0.9858863673652883
    --------------------------------------------------------

    deploy_weights/bisenet_2class_quchu_cloth-fp16-none-3_512_512-debug
    ----------------- Total Performance --------------------
    Overall Acc:     0.9900743026675991
    Mean Acc :       0.9875126713101177
    FreqW Acc :      0.9803651276458463
    Mean IoU :       0.9767595299768057
    Overall F1:      0.9882215794861182
    ----------------- Class IoU Performance ----------------
    background      : 0.967632757495317
    all_in_one_except_cloth : 0.9858863024582945
    --------------------------------------------------------

    deploy_weights/bisenet_2class_quchu_cloth-int8-kl_divergence-3_512_512-debug
    ----------------- Total Performance --------------------
    Overall Acc:     0.9900391853510482
    Mean Acc :       0.9876534861212527
    FreqW Acc :      0.9802999683614817
    Mean IoU :       0.9766867605825758
    Overall F1:      0.9881842393144329
    ----------------- Class IoU Performance ----------------
    background      : 0.9675407247922888
    all_in_one_except_cloth : 0.9858327963728626
    --------------------------------------------------------
    ```
    </details>
