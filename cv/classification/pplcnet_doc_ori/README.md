
# PP-LCNet_x1_0_doc_ori

## Model Arch

`PP-LCNet_x1_0_doc_ori`模型是基于`PP-LCNet_x1_0`微调的`文档图像方向`分类模型，含有四个类别，即0度，90度，180度，270度。

- [文档图像方向分类模块使用教程](https://www.paddleocr.ai/latest/version3.x/module_usage/doc_img_orientation_classification.html)
- [预处理](https://github.com/PaddlePaddle/PaddleX/blob/release/3.6/paddlex/inference/models/image_classification/predictor.py#L107)
- [原始权重](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar)
- [数据集](https://paddlepaddle.github.io/PaddleX/3.4/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#411-demo)

### pre-processing

`PP-LCNet_x1_0_doc_ori`网络的预处理操作可以按照如下步骤进行，即先对图片进行resize至256的尺寸，然后利用`CenterCrop`算子crop出224的图片对其进行归一化、减均值除方差等操作：

```python
[
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
]
```
> - 此模型实测，直接`resize=224`比`resize=256 -> crop=224`时的精度最佳
> - 具体应用场景可分别尝试获取最佳性能

### post-processing

`PP-LCNet_x1_0_doc_ori`系列网络的后处理操作是对网络输出进行softmax作为每个类别的预测值，然后根据预测值进行排序，选择topk作为输入图片的预测分数以及类别。

### backbone

- 参见：[cv/classification/pplcnet/README.md](../pplcnet/README.md)

### head

`PP-LCNet_x1_0_doc_ori`系列网络的head层由global-average-pooling层和一层 1x1 卷积层（等同于 FC 层），GAP 后的特征便不会直接经过分类层，而是先进行了融合，并将融合的特征进行分类。


## Build_In Deploy

- [pplcnet_x1_0_doc_ori.md](./source_code/pplcnet_x1_0_doc_ori.md)


## Tips
- 本模型：`文档图像方向`分类模型，含有四个类别，即0度，90度，180度，270度。（在文本定位之前使用）
- [pplcnet_textline_ori](../pplcnet_textline_ori/README.md)：`文本行方向`分类模型，含有两个类别，即0度，180度。（在文本定位之后使用）
