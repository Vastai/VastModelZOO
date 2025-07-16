## Deploy
### step.1 获取模型
- 从huggingface获取原始权重：[google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)

- 执行export_onnx.py导出onnx模型，参考[export_onnx.py](./export_onnx.py)。 为了避免报错， 使用如下命令进行onnxsim
    ```bash
    python export_onnx.py
    python -m onnxsim vit_base.onnx vit_base_sim.onnx
    ```

### step.2 获取数据集
- [校准数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [评估数据集](https://image-net.org/challenges/LSVRC/2012/index.php)
- [label_list](../../common/label/imagenet.txt)
- [label_dict](../../common/label/imagenet1000_clsid_to_human.txt)

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [vit_huggingface.yaml](../build_in/build/vit_huggingface.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 模型编译

    ```bash
    cd vision_transformer
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/vit_huggingface.yaml
    ```

### step.4 模型推理
1. runstream
    - 执行如下命令进行vsx推理， 参考[infer_vit.py](../build_in/vsx/infer_vit.py)
    ```bash
    python3 ../build_in/vsx/infer_vit.py \
        -m deploy_weights/vision_transformer_run_stream_fp16/mod \
        --norm_elf_file ../../common/elf/normalize \
        --space_to_depth_elf_file ../../common/elf/space_to_depth \
        --device_id 0 \
        --label_file /path/to/ImageNet/imagenet.txt \
        --dataset_filelist /path/to/ILSVRC2012_img_val_filelist.txt \
        --dataset_root /path/to/ImageNet/ 
        --dataset_output_file runstream_result.txt
    ```

    - 精度评估
    ```
    python ../../common/eval/eval_topk.py result.txt
    ```

    ```
    # fp16
    [VACC]:  top1_rate: 80.2 top5_rate: 95.402
    ```

### step.5 性能测试
1. 使用[infer_vit_prof.py](../build_in/vsx/infer_vit_prof.py)进行benchmark， 命令如下
    ```bash
    python3 vit_prof.py \
        -m vit-b-fp16-none-1_3_224_224-vacc/mod \
        --norm_elf_file ../../common/elf/normalize \
        --space_to_depth_elf_file ../../common/elf/space_to_depth \
        --device_ids [0] \
        --batch_size 1 \
        --instance 1 \
        --iterations 1024 \
        --shape [3,224,224] \
        --percentiles [50,90,95,99] \
        --input_host 1 \
        --queue_size 1

    ```

### appending
- 仅支持FP16
- 注onnxsim建议使用0.4.24
    ```bash
    # recommend
    onnxsim==0.4.24
    transformers==4.31.0
    ```

