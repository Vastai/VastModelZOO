## UIE

[UIE：Universal Information Extraction](https://arxiv.org/pdf/2203.12277)

## Code_Source
```
link：https://github.com/HUSTAI/uie_pytorch
branch: main
commit: 2eafcda44589144d2cb246b74e3bf2564ea6583f
```


## Model Arch

通用信息抽取统一框架UIE，实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。

<div align="center">
    <img src=../../../images/nlp/information_extraction/uie/arch.png height=400 hspace='10'/>
</div>

### UIE的优势

- 使用简单：用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。实现开箱即用，并满足各类信息抽取需求。

- 降本增效：以往的信息抽取技术需要大量标注数据才能保证信息抽取的效果，为了提高开发过程中的开发效率，减少不必要的重复工作时间，开放域信息抽取可以实现零样本（zero-shot）或者少样本（few-shot）抽取，大幅度降低标注数据依赖，在降低成本的同时，还提升了效果。

- 效果领先：开放域信息抽取在多种场景，多种任务上，均有不俗的表现。


## Build_In Deploy

### step.1 获取预训练模型
1. 克隆github仓库
    ```
    link：https://github.com/HUSTAI/uie_pytorch
    branch: main
    commit: 2eafcda44589144d2cb246b74e3bf2564ea6583f

    git clone https://github.com/HUSTAI/uie_pytorch.git

    ```

2. 原始模型基于paddlepaddle框架训练，需要转换为pytorch格式，参考：[convert.py](https://github.com/HUSTAI/uie_pytorch/blob/main/convert.py)
    > 此脚本将自动下载原始权重
    ```bash
    python convert.py \
    --input_model uie-base \
    --output_model uie_base_pytorch 
    ```

3. pytorch导出onnx：[export_model.py](https://github.com/HUSTAI/uie_pytorch/blob/main/export_model.py)
    ```bash
    mkdir -p export
    python export_model.py \
    --model_path ./uie_base_pytorch/ \
    --output_path ./export
    ```

4. 固定输入shape，并简化onnx: 
    ```bash
    python -m onnxsim inference.onnx uie_base.onnx \
    --input-shape input_ids:1,512 token_type_ids:1,512 attention_mask:1,512
    ```


### step.2 准备数据集
1. 下载原仓库数据集：[doccano_ext.json](https://bj.bcebos.com/paddlenlp/datasets/uie/doccano_ext.json)
2. 替换上述文件`doccano_ext.json`中的关键词"目的地"为"终点站"

3. 参考源仓库[数据标注](https://github.com/HUSTAI/uie_pytorch?tab=readme-ov-file#42-数据标注)，对数据集进行数据转换：[doccano.py](https://github.com/HUSTAI/uie_pytorch/blob/main/doccano.py)
    > 执行后会在`--save_dir`目录下生成训练/验证/测试集文件
  
    ```bash
    python doccano.py \
        --doccano_file ./doccano_ext.json \
        --task_type ext \
        --save_dir ./data \
        --splits 0.8 0.2 0
    ```

### step.3 模型转换
1. 根据具体模型修改配置文件
    - [hustai_uie_base.yaml](./build_in/build/hustai_uie_base.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`

2. 模型编译
    ```bash
    vamc compile ./build_in/build/hustai_uie_base.yaml
    ```
    > 当前只支持fp16

### step.4 模型推理
- 根据具体模型修改执行参数路径
- 推理示例，参考：[uie_vsx_infer.py](./build_in/vsx/python/uie_vsx_infer.py)
  ```bash
  export GLOG_minloglevel=1
  python3 build_in/vsx/python/uie_vsx_infer.py \
      --model_prefix deploy_weights/uie_base_fp16/mod \
      --vdsp_params  build_in/vdsp_params/hustai-uie_base-vdsp_params.json \
      --tokenizer_path path/to/uie_base_pytorch/ \
      --batch_size  64 \
      --device 0 \
      --schema "航母" \
      --input_txt "印媒所称的“印度第一艘国产航母”—“维克兰特”号"
  
  # 预期输出
  # [{"'航母'": [{'text': '维克兰特”号', 'start': 18, 'end': 24, 'probability': np.float16(0.6504)}]}]
  ```

### step.5 精度测试
- 精度评估示例，参考：[uie_vsx_evaluate.py](./build_in/vsx/python/uie_vsx_evaluate.py)

  ```bash
  export GLOG_minloglevel=1
  python build_in/vsx/python/uie_vsx_evaluate.py \
    --model_prefix deploy_weights/uie_base_fp16/mod \
    --vdsp_params  build_in/vdsp_params/hustai-uie_base-vdsp_params.json \
    --tokenizer_path path/to/uie_base_pytorch/ \
    -d 0 \
    --batch_size 16 \
    --test_path path/to/dev.txt

    # 预期精度统计结果                                                                 
    # Class Name: all_classes
    # Evaluation Precision: 0.60714 | Recall: 0.51515 | F1: 0.55738

    # pytorch模型推理bench精度
    # Evaluation Precision: 0.60714 | Recall: 0.51515 | F1: 0.55738
  ```

### step.6 性能测试
- 性能评估示例，参考：[uie_vsx_prof.py](./build_in/vsx/python/uie_vsx_prof.py)

  ```bash
  export GLOG_minloglevel=1

  # 测试模型最大吞吐
  python3 build_in/vsx/python/uie_vsx_prof.py \
  --model_prefix deploy_weights/uie_base_fp16/mod \
  --vdsp_params  build_in/vdsp_params/hustai-uie_base-vdsp_params.json \
  --tokenizer_path path/to/uie_base_pytorch/ \
  --device_ids  [0] \
  --batch_size  64 \
  --instance 1 \
  --iterations 20 \
  --percentiles "[50,90,95,99]" \
  --input_host 1 \
  --queue_size 1

  # 测试模型最小时延
  python3 build_in/vsx/python/uie_vsx_prof.py \
  --model_prefix deploy_weights/uie_base_fp16/mod \
  --vdsp_params  build_in/vdsp_params/hustai-uie_base-vdsp_params.json \
  --tokenizer_path path/to/uie_base_pytorch/ \
  --device_ids  [0] \
  --batch_size  1 \
  --instance 1 \
  --iterations 1500 \
  --percentiles "[50,90,95,99]" \
  --input_host 1 \
  --queue_size 0
  ```
