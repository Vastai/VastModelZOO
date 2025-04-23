## mmyolo yolov5

### step.1 获取预训练模型

1. vacc暂不支持mmyolo库算法后处理，因此模型转换时需进行截断。对于yolov5算法来说，可以通过修改`YOLOv5HeadModule`模块截断后处理，代码修改具体如下

    <details><summary>点击查看代码细节</summary>

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

    </details>

2. mmyolo模型转换可以通过mmdeploy工具转换，也可通过mmyolo调用模型进行转换。这里给出通过修改mmyolo中`image_demo.py`脚本调用模型进行转换的脚本，如下

    <details><summary>点击查看代码细节</summary>

    ```python
    # Copyright (c) OpenMMLab. All rights reserved.
    import os
    from argparse import ArgumentParser

    import mmcv
    from mmdet.apis import inference_detector, init_detector
    from mmengine.logging import print_log
    from mmengine.utils import ProgressBar

    from mmyolo.registry import VISUALIZERS
    from mmyolo.utils import register_all_modules, switch_to_deploy
    from mmyolo.utils.misc import get_file_list


    def parse_args():
        parser = ArgumentParser()
        parser.add_argument(
            'img', help='Image path, include image file, dir and URL.')
        parser.add_argument('config', help='Config file')
        parser.add_argument('checkpoint', help='Checkpoint file')
        parser.add_argument(
            '--out-dir', default='./output', help='Path to output file')
        parser.add_argument(
            '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument(
            '--show', action='store_true', help='Show the detection results')
        parser.add_argument(
            '--deploy',
            action='store_true',
            help='Switch model to deployment mode')
        parser.add_argument(
            '--score-thr', type=float, default=0.3, help='Bbox score threshold')
        args = parser.parse_args()
        return args


    def main():
        args = parse_args()

        # register all modules in mmdet into the registries
        register_all_modules()

        # build the model from a config file and a checkpoint file
        model = init_detector(args.config, args.checkpoint, device=args.device)
        print(model)
        import torch
        import thop
        from thop import clever_format
        
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


        if args.deploy:
            switch_to_deploy(model)

        if not os.path.exists(args.out_dir) and not args.show:
            os.mkdir(args.out_dir)

        # init visualizer
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta

        # get file list
        files, source_type = get_file_list(args.img)

        # start detector inference
        progress_bar = ProgressBar(len(files))
        for file in files:
            result = inference_detector(model, file)

            img = mmcv.imread(file)
            img = mmcv.imconvert(img, 'bgr', 'rgb')

            if source_type['is_dir']:
                filename = os.path.relpath(file, args.img).replace('/', '_')
            else:
                filename = os.path.basename(file)
            out_file = None if args.show else os.path.join(args.out_dir, filename)

            visualizer.add_datasample(
                os.path.basename(out_file),
                img,
                data_sample=result,
                draw_gt=False,
                show=args.show,
                wait_time=0,
                out_file=out_file,
                pred_score_thr=args.score_thr)
            progress_bar.update()

        if not args.show:
            print_log(
                f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


    if __name__ == '__main__':
        main()

    ```

    </details>

3. 通过运行如下脚本即可生成onnx与torchscript模型文件

    ```bash
    python demo/image_demo.py ~/project/yolov5/data/images/bus.jpg configs/yolov5/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco.py models/yolov5/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_234308-7a2ba6bf.pth
    ```

### step.2 准备数据集
- [校准数据集](http://images.cocodataset.org/zips/val2017.zip)
- [评估数据集](http://images.cocodataset.org/zips/val2017.zip)
- [gt: instances_val2017.json](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [label: coco.txt](../../common/label/coco.txt)


### step.3 模型转换

1. 参考瀚博训推软件生态链文档，获取模型转换工具: [vamc v3.0+](../../../../docs/vastai_software.md)

2. 根据具体模型，修改编译配置
    - [mmyolo_yolov5.yaml](../build_in/build/mmyolo_yolov5.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

3. 模型编译

    ```bash
    cd yolov5
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/mmyolo_yolov5.yaml
    ```

### step.4 模型推理

1. 参考瀚博训推软件生态链文档，获取模型推理工具：[vaststreamx v2.8+](../../../../docs/vastai_software.md)

2. runstream推理：[detector_mmyolo.py](../build_in/vsx/detector_mmyolo.py)
    - 配置模型路径和测试数据路径等参数

    ```
    python ../build_in/vsx/detector_mmyolo.py \
        --file_path path/to/det_coco_val \
        --model_prefix_path deploy_weights/mmyolo_yolov5s_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/mmyolo-yolov5s-vdsp_params.json \
        --label_txt path/to/coco.txt \
        --save_dir ./runstream_output \
        --device 0
    ```

    - 精度评估，参考：[eval_map.py](../../common/eval/eval_map.py)
    ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt ./runstream_output
    ```

    <details><summary>点击查看精度测试结果</summary>
    
    ```
    # 模型名：yolov5s-640

    # fp16
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.368
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.562
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.398
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.200
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.416
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.484
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.295
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.474
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.512
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.312
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.568
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.645
    {'bbox_mAP': 0.368, 'bbox_mAP_50': 0.562, 'bbox_mAP_75': 0.398, 'bbox_mAP_s': 0.2, 'bbox_mAP_m': 0.416, 'bbox_mAP_l': 0.484, 'bbox_mAP_copypaste': '0.368 0.562 0.398 0.200 0.416 0.484'}

    # int8
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.546
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.376
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.184
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.389
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.461
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.280
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.453
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.492
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.295
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.543
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.623
    {'bbox_mAP': 0.347, 'bbox_mAP_50': 0.546, 'bbox_mAP_75': 0.376, 'bbox_mAP_s': 0.184, 'bbox_mAP_m': 0.389, 'bbox_mAP_l': 0.461, 'bbox_mAP_copypaste': '0.347 0.546 0.376 0.184 0.389 0.461'}
    ```

    </details>


### step.5 性能精度
1. 参考瀚博训推软件生态链文档，获取模型性能测试工具：[vamp v2.4+](../../../../docs/vastai_software.md)

2. 性能测试
    - 配置[mmyolo-yolov5s-vdsp_params.json](../build_in/vdsp_params/mmyolo-yolov5s-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/mmyolo_yolov5s_run_stream_int8/mod --vdsp_params ../build_in/vdsp_params/mmyolo-yolov5s-vdsp_params.json -i 1 p 1 -b 1 -d 0
    ```

3. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path path/to/coco_val2017 \
        --target_path  path/to/coco_val2017_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理获取npz结果输出
    ```bash
    vamp -m deploy_weights/mmyolo_yolov5s_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/mmyolo-yolov5s-vdsp_params.json \
        -i 1 p 1 -b 1 \
        --datalist datasets/coco_npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz文件，参考：[npz_decode.py](../../common/utils/npz_decode.py)
    ```bash
    python ../../common/utils/npz_decode.py \
        --txt result_npz --label_txt datasets/coco.txt \
        --input_image_dir datasets/coco_val2017 \
        --model_size 640 640 \
        --vamp_datalist_path datasets/coco_npz_datalist.txt \
        --vamp_output_dir npz_output
    ```

    - 精度统计，参考：[eval_map.py](../../common/eval/eval_map.py)
    ```bash
    python ../../common/eval/eval_map.py \
        --gt path/to/instances_val2017.json \
        --txt path/to/vamp_draw_output
    ```