### step.1 获取预训练模型
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

2. segment的reshape在build时仍有问题，同样需要截断， 修改[Segment](https://github.com/ultralytics/ultralytics/blob/b1119d512e738e90f2327b316216b069ed576a56/ultralytics/nn/modules/head.py#L74)类如下
    <details><summary>head-segment</summary>

    ```bash
    class Segment(Detect):
        """YOLOv8 Segment head for segmentation models."""

        def __init__(self, nc=80, nm=32, npr=256, ch=()):
            """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
            super().__init__(nc, ch)
            self.nm = nm  # number of masks
            self.npr = npr  # number of protos
            self.proto = Proto(ch[0], self.npr, self.nm)  # protos
            self.detect = Detect.forward

            c4 = max(ch[0] // 4, self.nm)
            self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

        def forward(self, x):
            """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
            p = self.proto(x[0])  # mask protos
            bs = p.shape[0]  # batch size
            tmp_mc = []
            for i in range(self.nl):
                tmp_mc.append(self.cv4[i](x[i]))
            # mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
            x = self.detect(self, x)
            if self.training:
                return x, tmp_mc, p
            return x+tmp_mc+ [p]
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
    model = YOLO('yolov8s-seg.pt')  # load a custom trained
    ### Export the model
    model.export(format='onnx', imgsz= 640, simplify = True, opset=11, dynamic=True)
    ```

### step.2 准备数据集
该模型使用coco2017数据集，请到coco官网自行下载coco2017，针对`int8`校准数据可从该数据集中任选50张作为校准数据集，[coco2017](https://cocodataset.org/#download)

```
├── COCO
|   ├── val
|   |    ├── 000000000139.jpg
│   |    ├── 000000000285.jpg
│   |    ├── ......
|   ├── instances_val2017.json
```

```bash
# label.txt
person
bicycle
car
motorcycle
airplane
bus
train
```

### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[yolov8_seg.yaml](../vacc_code/build/yolov8_seg.yaml)：
    ```bash
    vamc build ../vacc_code/build/yolov8_seg.yaml
    ```

### step.4 benchmark
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    ```bash
    vamp -m deploy_weights/yolov8s_seg-int8-percentile-3_640_640-vacc/yolov8s_seg --vdsp_params ../vacc_code/vdsp_params/ultralytics-yolov8s_seg-vdsp_params.json -i 2 p 2 -b 1
    ```
4. npz结果输出
    ```bash
    vamp -m deploy_weights/yolov8s_seg-int8-percentile-3_640_640-vacc/yolov8s_seg --vdsp_params ../vacc_code/vdsp_params/ultralytics-yolov8s_seg-vdsp_params.json -i 2 p 2 -b 1 --datalist npz_datalist.txt --path_output npz_output
    ```
5. [npz_decode.py](../vacc_code/runmodel/npz_decode.py)，解析vamp输出的npz文件，生成predictions.json
    ```bash
    python ../vacc_code/runmodel/npz_decode.py  --label-txt coco.txt --input-image datasets/coco_val2017 --model_size 640 640 --datalist-txt datasets/npz_datalist.txt --vamp-output npz_output
    ```
6. [eval_map.py](../vacc_code/runmodel/eval.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../vacc_code/runmodel/eval.py --gt path/to/instances_val2017.json --txt TEMP/predictions.json
   ```