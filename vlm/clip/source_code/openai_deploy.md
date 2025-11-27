## Deploy
### step.1 获取模型
- code source如下， 使用clip api在下一步导出onnx自动下载模型
    ```bash
    # source
    https://github.com/openai/CLIP/tree/main
    # branch
    main
    # commit
    a1d071733d711
    ```

### step.2 onnx导出
1. pip install clip or pip install git+https://github.com/openai/CLIP.git
2. modify：导出image backbone， 修改`clip/model.py/`的[forward函数](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L358)如下
    ```python
    def forward(self, inputs):
        if len(inputs.size())>2:
            image_features = self.encode_image(inputs)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            return image_features
        else:
            text_features = self.encode_text(inputs)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            return text_features
    ```
3. 执行命令导出onnx, 参考[export_onnx.py](./export_onnx.py)
    ```bash
    mv ./export_onnx.py /path/to/CLIP && cd /path/to/CLIP
    # export image backbone
    python export_onnx.py --onnx_file onnx/images.onnx --export image
    # export text backbone
    python export_onnx.py --onnx_file onnx/text.onnx --export text
    ```

### step.3 获取数据集
> 以ISLVRC2012来检验clip在图像分类领域的能力
- [校准数据集](https:/image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https:/image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../../cv/classification/common/label/imagenet.txt)
- [label_dict](../../../cv/classification/common/label/imagenet1000_clsid_to_human.txt)

### step.4 模型转换

1. 根据具体模型，修改编译配置
    - 当前模型只支持fp16
    - [clip_image_openai.yaml](../build_in/build/clip_image_openai.yaml)
    - [clip_text_openai.yaml](../build_in/build/clip_text_openai.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 模型编译
    ```bash
    cd clip
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/clip_image_openai.yaml
    vamc compile ../build_in/build/clip_text_openai.yaml
    ```

### step.5 模型推理

1. runstream
    - [elf文件](../../../cv/classification/common/elf/)
    - 参考：[infer_clip.py](../build_in/vsx/python/infer_clip.py)
    ```bash
    python3 ../build_in/vsx/python/infer_clip.py \
        --imgmod_prefix deploy_weights/clip_image_openai_run_stream_fp16/mod \
        --norm_elf /path/to/elf/normalize \
        --space2depth_elf /path/to/elf/space_to_depth \
        --txtmod_prefix deploy_weights/clip_text_openai_run_stream_fp16/mod \
        --txtmod_vdsp_params ../build_in/vdsp_params/openai-clip-vdsp_params.json \
        --label_file /path/to/imagenet.txt \
        --device_id 0 \
        --dataset_root /path/to/ILSVRC2012_img_val \
        --dataset_output_file clip_result.txt
    ```

    - 精度评估，参考：[eval_topk.py](../../../cv/classification/common/eval/eval_topk.py)
    ```bash
    python ../../../../cv/classification/common/eval/eval_topk.py clip_result.txt
    ```

    ```
    [VACC]:  top1_rate: 55.624 top5_rate: 82.598
    ```

### step.6 性能测试

1. 参考[infer_clip_image_prof.py](../build_in/vsx/python/infer_clip_image_prof.py)测试clip_image的性能， 命令如下
    ```bash
    python3 ../build_in/vsx/python/infer_clip_image_prof.py \
        -m deploy_weights/clip_image_openai_run_stream_fp16/mod \
        --norm_elf /path/to/elf/normalize \
        --space2depth_elf /path/to/elf/space_to_depth \
        --device_ids  [0] \
        --batch_size  1 \
        --instance 1 \
        --iterations 1000 \
        --percentiles "[50,90,95,99]" \
        --input_host 1 \
        --queue_size 1

    ```

2. 参考[infer_clip_text_prof.py](../build_in/vsx/python/infer_clip_text_prof.py)测试clip_image的性能， 命令如下
    ```bash
    python3 ../build_in/vsx/python/infer_clip_text_prof.py \
        -m deploy_weights/clip_text_openai_run_stream_fp16/mod \
        --vdsp_params ../build_in/vdsp_params/openai-clip-vdsp_params.json \
        --device_ids  [0] \
        --batch_size  1 \
        --instance 1 \
        --iterations 1500 \
        --percentiles "[50,90,95,99]" \
        --input_host 1 \
        --queue_size 1
    ```

### appending
- 如果遇到报错`module 'clip' has no attribute 'tokenize'`, 可使用源码安装clip
- 主要限制pip依赖库版本，否则可能转换报错：
    ```
    onnx==1.16.2
    onnxruntime==1.19.2
    onnxsim==0.4.36
    torch==1.8.0
    torchvision==0.9.0
    ```