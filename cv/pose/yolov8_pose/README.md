
<img width="100%" src="../../../images/cv/pose/yolov8_pose/task.jpg"></a>

# Yolov8_pose

## Code Source
```
link: https://github.com/ultralytics/ultralytics
branch: main
commit: b1119d512e738
```

## Model Arch

### pre-processing

yolov8_pose系列的预处理主要是对输入图片利用`letterbox`算子进行resize，然后进行归一化

### post-processing

yolov8_pose系列的后处理是首先进行box decode之后进行nms,box head部分和yolov5一致，pose head部分根据框索引查询N* 51的关键点信息(17*3)，然后进行letterbox的反操作放缩到原图之上。关键点信息包含xy坐标以及表示是否可见的score，当score大于0.5时表示该关键点在图片可见。

### backbone

Yolov8 backbone和Neck部分参考了YOLOv7 ELAN设计思想，将YOLOv5的C3结构换成了梯度流更丰富的C2f结构，并对不同尺度模型调整了不同的通道数，属于对模型结构精心微调，不再是无脑一套参数应用所有模型，大幅提升了模型性能

### head

yolov8_pose det Head部分和yolov8检测部分类似。det head相比YOLOv5改动较大，换成了目前主流的解耦头结构，将分类和检测头分离，同时也从Anchor-Based换成了Anchor-Free，Loss计算方面采用了TaskAlignedAssigner正样本分配策略，并引入了 Distribution Focal Loss。pose head使用空间金字塔为每个检测到的对象预测17个关键点。

### common

- C2f
- SPPF
- letterbox
- DFL

## Model Info

### 模型性能

| 模型  | 源码 | mAP@.5 | mAP@.5:.95 | flops(B) | params(M) | input size |
| :---: | :--: | :--: | :--: | :---: | :----: | :--------: |
| yolov8n-pose |[ultralytics](https://github.com/ultralytics/ultralytics)|   80.1    |   50.4   |   9.2    |    3.3    |        640    |
| yolov8s-pose |[ultralytics](https://github.com/ultralytics/ultralytics)|   86.2    |   60.0   |   30.2    |    11.6   |        640    |
| yolov8m-pose |[ultralytics](https://github.com/ultralytics/ultralytics)|   88.8    |   65.8   |   81.0   |    26.4    |        640    |
| yolov8l-pose |[ultralytics](https://github.com/ultralytics/ultralytics)|   90.0    |   67.6  |   168.6    |    44.4    |        640    |
| yolov8x-pose |[ultralytics](https://github.com/ultralytics/ultralytics)|   90.2    |   69.2   |  263.2   |    69.4   |        640    |
| yolov8x-pose-p6 |[ultralytics](https://github.com/ultralytics/ultralytics)|   91.2  |   71.6   |  1066.4   |    99.1    |        1280    |

### 测评数据集说明

![](../../../images/dataset/coco.png)


[MS COCO](https://cocodataset.org/#download)的全称是Microsoft Common Objects in Context，是微软于2014年出资标注的Microsoft COCO数据集，与ImageNet竞赛一样，被视为是计算机视觉领域最受关注和最权威的比赛数据集之一。 

COCO数据集支持目标检测、关键点检测、实例分割、全景分割与图像字幕任务。

### 评价指标说明

- mAP: mean of Average Precision, 多类别的AP的平均值；AP即平均精度，是Precision-Recall曲线下的面积
- mAP@.5: 即将IoU设为0.5时，计算每一类的所有图片的AP，然后所有类别求平均，即mAP
- mAP@.5:.95: 表示在不同IoU阈值（从0.5到0.95，步长0.05）上的平均mAP

## Build_In Deploy

### step.1 模型准备
1. 目前Compiler暂不支持四维softmax算子，yolov8中DFL模块包含四维softmax算子，但是由于其后的卷积层不参与训练，因此可以将该算子后的处理截断写在host侧。综上，转换模型时可以修改[Detect](https://github.com/ultralytics/ultralytics/blob/b1119d512e738e90f2327b316216b069ed576a56/ultralytics/nn/modules/head.py#L22)类如下
    <details><summary>head-detect</summary>

    ```python
    class Detect(nn.Module):
        """YOLOv8 Detect head for detection models."""
        dynamic = False  # force grid reconstruction
        export = False  # export mode
        shape = None
        anchors = torch.empty(0)  # init
        strides = torch.empty(0)  # init

        def __init__(self, nc=80, ch=()):  # detection layer
            super().__init__()
            self.nc = nc  # number of classes
            self.nl = len(ch)  # number of detection layers
            self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
            self.no = nc + self.reg_max * 4  # number of outputs per anchor
            self.stride = torch.zeros(self.nl)  # strides computed during build
            c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        def forward(self, x):
            """Concatenates and returns predicted bounding boxes and class probabilities."""
            shape = x[0].shape  # BCHW
            
            ## export onnx
            y = []
            for i in range(self.nl):
                y.append(self.cv2[i](x[i]))
                y.append(self.cv3[i](x[i]))
            return y

        def bias_init(self):
            """Initialize Detect() biases, WARNING: requires stride availability."""
            m = self  # self.model[-1]  # Detect() module
            # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
            # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
            for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    ```
    </details>

2. pose的reshape在build时仍有问题，同样需要截断， 修改[Pose](https://github.com/ultralytics/ultralytics/blob/b1119d512e738e90f2327b316216b069ed576a56/ultralytics/nn/modules/head.py#L100)类如下
    <details><summary>head-pose</summary>

    ```python
    class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

        def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
            """Initialize YOLO network with default parameters and Convolutional Layers."""
            super().__init__(nc, ch)
            self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
            self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
            self.detect = Detect.forward

            c4 = max(ch[0] // 4, self.nk)
            self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

        
        def forward(self, x):
            """Perform forward pass through YOLO model and return predictions."""
            bs = x[0].shape[0]  # batch size
            kpt = []
            for i in range(self.nl):
                kpt.append(self.cv4[i](x[i]))
            x = self.detect(self, x)
            if self.training:
                return x, kpt
            return x + kpt

        def kpts_decode(self, bs, kpts):
            """Decodes keypoints."""
            ndim = self.kpt_shape[1]
            if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
                if ndim == 3:
                    a = torch.cat((a, y[:, :, 1:2].sigmoid()), 2)
                return a.view(bs, self.nk, -1)
            else:
                y = kpts.clone()
                if ndim == 3:
                    y[:, 2::3].sigmoid_()  # inplace sigmoid
                y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
                y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
                return y
    ```
    </details>

3. 转换onnx时，需指定batch为动态，可以修改[export](https://github.com/ultralytics/ultralytics/blob/b1119d512e738e90f2327b316216b069ed576a56/ultralytics/yolo/engine/exporter.py#L292)代码
    <details><summary>exporter</summary>

    ```python
        @try_export
        def export_onnx(self, prefix=colorstr('ONNX:')):
            """YOLOv8 ONNX export."""
            requirements = ['onnx>=1.12.0']
            if self.args.simplify:
                requirements += ['onnxsim>=0.4.17', 'onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime']
            check_requirements(requirements)
            import onnx  # noqa

            opset_version = self.args.opset or get_latest_opset()
            LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...')

            f = str(self.file.with_suffix('.onnx'))
            f = f.replace('-', '_')
            f = f.replace('.onnx', '-'+str(self.im.shape[-1])+'.onnx')

            output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
            dynamic = self.args.dynamic
            if dynamic:
                dynamic = {'images': {0: '-1'}}
                if isinstance(self.model, SegmentationModel):
                    dynamic['output0'] = {0: '-1'}
                    dynamic['output1'] = {0: '-1'}
                elif isinstance(self.model, DetectionModel):
                    dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

            torch.onnx.export(
                self.model.cpu() if dynamic else self.model,  # --dynamic only compatible with cpu
                self.im.cpu() if dynamic else self.im,
                f,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
                input_names=['images'],
                output_names=output_names,
                dynamic_axes=dynamic or None)

            # Checks
            model_onnx = onnx.load(f)  # load onnx model
            # onnx.checker.check_model(model_onnx)  # check onnx model

            # Simplify
            if self.args.simplify:
                try:
                    import onnxsim

                    LOGGER.info(f'{prefix} simplifying with onnxsim {onnxsim.__version__}...')
                    # subprocess.run(f'onnxsim {f} {f}', shell=True)
                    model_onnx, check = onnxsim.simplify(model_onnx)
                    assert check, 'Simplified ONNX model could not be validated'
                except Exception as e:
                    LOGGER.info(f'{prefix} simplifier failure: {e}')

            # Metadata
            for k, v in self.metadata.items():
                meta = model_onnx.metadata_props.add()
                meta.key, meta.value = k, str(v)

            onnx.save(model_onnx, f)
            return f, model_onnx
    ```
    </details>
4. onnx导出
    ```python
    from ultralytics import YOLO
    ### load a custom trained
    model = YOLO('yolov8s-pose.pt')  # load a custom trained
    ### Export the model
    model.export(format='onnx', imgsz= 640, simplify = True, opset=11, dynamic=True)
    ```

### step.2 准备数据集
> 注意标签文件为 person_keypoints_val2017
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: person_keypoints_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../detection/common/label/coco.txt)


### step.3 模型转换

1. 根据具体模型修改配置文件
    -[ultralytics_yolov8_pose.yaml](./build_in/build/ultralytics_yolov8_pose.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    - 注意需要先替换yaml文件中校正集数据的路径
    ```bash
    cd yolov8_pose
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/ultralytics_yolov8_pose.yaml
    ```
    - 转换后将在当前目录下生成`deploy_weights/ultralytics_yolov8_pose_int8`文件夹，其中包含转换后的模型文件。

### step.4 模型推理
1. 参考[vsx脚本](./build_in/vsx/python/yolov8_pose_vsx.py)，修改参数并运行如下脚本
    ```bash
    python ../build_in/vsx/python/yolov8_pose_vsx.py \
        --file_path  /path/to/coco/det_coco_val \
        --model_prefix_path deploy_weights/ultralytics_yolov8_pose_int8/mod \
        --vdsp_params_info ../build_in/vdsp_params/ultralytics-yolov8s_pose-vdsp_params.json \
        --label_txt /path/to/coco.txt \
        --save_dir ./infer_output \
        --device 0
    ```
    - 注意替换命令行中--file_path为实际路径

2. [eval.py](./source_code/eval.py)，精度统计，指定gt路径和上步骤中的txt保存路径，即可获得精度指标
    ```
    python ../source_code/eval.py --gt path/to/person_keypoints_val2017.json --pred infer_output/predictions.json 
    ```
    - 测试精度如下：
    ```
    Evaluate annotation type *keypoints*
    DONE (t=5.67s).
    Accumulating evaluation results...
    DONE (t=0.21s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.571
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.847
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.629
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.504
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.679
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.643
    Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.895
    Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.704
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.564
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.754
    ```

### step.5 性能测试
1. 基于[image2npz.py](../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    
    ```bash
    python ../../common/utils/image2npz.py  --dataset_path /path/to/eval/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```

2. 性能测试
    ```bash
    vamp -m deploy_weights/ultralytics_yolov8_pose_int8/mod --vdsp_params ../build_in/vdsp_params/ultralytics-yolov8s_pose-vdsp_params.json -i 2 p 2 -b 1
    ```