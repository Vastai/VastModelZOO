### step.1 获取预训练模型
1. [line355](https://github.com/starhiking/HeatmapInHeatmap/blob/7174ea28ebf5846d1406d06c665fc09f06630022/lib/model/get_hourglass.py#L355C1-L355C1) 插入`return out_preds[-1], offset`, 修改如下：

    ```python
    def forward(self,x):
        x = self.pre_conv(x)
        out_preds = []
        # out_offsets = []

        for i in range(self.config.num_stack):
            hg = eval('self.hg'+str(i))(x)
            ll = eval('self.hg'+str(i)+'_res1')(hg)
            feature = eval('self.hg'+str(i)+'_lin1')(ll)
            preds = eval('self.hg'+str(i)+'_conv_pred')(feature)
            out_preds.append(preds)

            # if self.offset_func is not None:
            #     offset = eval('self.offset_' + str(i))(x,hg,preds)
            #     out_offsets.append(offset)

            # if i < self.config.num_stack - 1:
            merge_feature = eval('self.hg'+str(i)+'_conv1')(feature)
            merge_preds = eval('self.hg'+ str(i)+'_conv2')(preds)
            x = x+merge_feature+merge_preds
        
        if self.offset_func is not None:
            offset = self.offset_head(x)
        ### export
        return out_preds[-1], offset

    ```
2. 将[export.py](./export.py)移动到tools目录下， 执行如下命令导出onnx
    ```bash
    python tools/export.py --config_file experiments/Data_WFLW/HIHC_64x8_hg_l2.py --resume_checkpoint WFLW.pth
    ```
### step.2 准备数据集
- [校准数据集](https://wywu.github.io/projects/LAB/WFLW.html)
- [评估数据集](https://wywu.github.io/projects/LAB/WFLW.html)
    - 需要自己生成预处理后的数据，进入[工程](https://github.com/jhb86253817/PIPNet.git)，按如下步骤操作：
    ```bash
    cd lib
    #执行预处理脚本时需按照实际模型输入尺寸进行修改，本例中为256
    python preprocess.py WFLW
    ```

### step.3 模型转换

1. 根据具体模型，修改编译配置
    - [official_hih.yaml](../build_in/build/official_hih.yaml)
    
    > - 编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译

    ```bash
    cd hih
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_hih.yaml
    ```

### step.4 模型推理

- 参考：[vsx_infer.py](../build_in/vsx/python/vsx_infer.py)
    ```bash
    python ../build_in/vsx/python/vsx_infer.py \
        --file_path  /path/to/face_alignment/wflw/WFLW  \
        --config_path ../source_code/base_config.yaml \
        --model_weight_path deploy_weights/official_hih_fp16/  \
        --model_name mod \
        --vdsp_params_info ../build_in/vdsp_params/official-hih_wflw_4stack-vdsp_params.json \
        --save_dir ./infer_output \
        --device 0
    ```

    ```
    # fp16
    NME %: 4.265816700826202
    FR_0.1% : 4.120000000000001
    AUC_0.1: 0.5978636

    # int8
    NME %: 4.261466921153629
    FR_0.1% : 4.079999999999995
    AUC_0.1: 0.5973917333333334
    ```

### step.5 性能精度测试
1. 性能测试
    配置[official-hih_wflw_4stack-vdsp_params.json](../build_in/vdsp_params/official-hih_wflw_4stack-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_hih_int8/mod --vdsp_params ../build_in/vdsp_params/official-hih_wflw_4stack-vdsp_params.json -i 2 p 2 -b 2
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；

    - 数据准备，基于[image2npz.py](../../common/util/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/util/image2npz.py \
        --dataset_path data/WFLW/images_test \
        --target_path  path/to/vamp_test_data \
        --text_path npz_datalist.txt
    ```

    - vamp推理获取npz结果输出
    ```bash
    vamp -m deploy_weights/deploy_weightsofficial_hih_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-hih_wflw_4stack-vdsp_params.json \
        -i 1 p 1 -b 1 \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析vamp输出的npz文件，并得到精度结果[npz_decode.py](../build_in/npz_decode.py)
    ```bash
    python ../build_in/npz_decode.py  -input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output \
        --input_shape 256 256 \
        --vamp_flag
    ```