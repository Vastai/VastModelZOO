## pan_pp.pytorch

```
link: https://github.com/whai362/pan_pp.pytorch/tree/master/config/pan
branch: master
commit: 674e4d8c88635543e803e4dae4f992e1cc7ea645
```

### step.1 获取预训练模型


1. 修改[pan forward](https://github.com/whai362/pan_pp.pytorch/blob/master/models/pan.py)输出

    ```python
    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                img_metas=None,
                cfg=None):
        if cfg is None:
            cfg = EasyDict()
            cfg['report_speed'] = False

        # xxxxxxxxxxx

         if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels,
                                          training_masks, gt_instances,
                                          gt_bboxes)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), 4)
            print("----forward output shape------:",det_out.shape)
            return det_out
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)
        return outputs
    ```
2. export

   ```bash
    python ../source_code/export.py config/pan/pan_r18_ctw.py ./weights/pan/pan_r18_ctw.pth.tar
   ```

### step.2 准备数据集
- 下载[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)数据集，通过[image2npz.py](../../common/utils/image2npz.py)，转换为对应npz文件


### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [pytorch_pan.yaml](../build_in/build/pytorch_pan.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd pan
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/pytorch_pan.yaml
    ```

### step.4 模型推理

0. 运行模型后处理前，需编译后处理插件
    ```bash
    cd ../source_code/post_process/pa/
    python setup.py build_ext --inplace
    ```

1. runstream
    - 参考：[vsx.py](../build_in/vsx/python/vsx.py)
    ```bash
    python ../build_in/vsx/python/vsx.py \
        --file_path  /path/to/ch4_test_images  \
        --model_prefix_path deploy_weights/pytorch_pan_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/pan_pp.pytorch-pan_r18_ic15-vdsp_params.json \
        --save_dir runstream_output \
        --device 0
    ```

    - 参考[eval](https://github.com/whai362/pan_pp.pytorch/tree/master/eval)，进行精度评估。需根据数据集选择相对应脚本
    ```
    cd text_detection_eval/ic15
    python3 script.py -g=gt.zip -s=../../runstream_output/submit_ic15.zip
    ```

    ```
    # fp16
    {"precision": 0.8380462724935732, "recall": 0.784785748675975, "hmean": 0.8105420188960717, "AP": 0}

    # int8
    {"precision": 0.8403141361256544, "recall": 0.7727491574386134, "hmean": 0.8051166290443942, "AP": 0}
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[pan_pp.pytorch-pan_r18_ic15-vdsp_params.json](../build_in/vdsp_params/pan_pp.pytorch-pan_r18_ic15-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/pytorch_pan_run_stream_int8/mod \
    --vdsp_params ../build_in/vdsp_params/pan_pp.pytorch-pan_r18_ic15-vdsp_params.json \
    -i 1 p 1 -b 1
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`：
    ```bash
    python ../../common/utils/image2npz.py \
        --dataset_path /path/to/lane_detection/Tusimple/clips \
        --target_path /path/to/lane_detection/Tusimple/clips_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/pytorch_pan_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/pan_pp.pytorch-pan_r18_ic15-vdsp_params.json \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析预测结果，参考：[npz_decode.py](../build_in/vdsp_params/npz_decode.py)

    ```bash
    python ../build_in/vdsp_params/npz_decode.py --gt_dir  /path/to//ocr/ICDAR2015/ch4_test_images --input_npz_pathnpz_datalist.txt --out_npz_dir npz_out --save_dir output_npz
    ```
    
    - 精度评估，参考[eval](https://github.com/whai362/pan_pp.pytorch/tree/master/eval)，进行精度评估。需根据数据集选择相对应脚本
    ```
    cd text_detection_eval/ic15
    python3 script.py -g=gt.zip -s=../../runstream_output/submit_ic15.zip
    ```
