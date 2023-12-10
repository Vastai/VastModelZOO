### step.1 获取预训练模型
```bash
cd mmyolo
python ./projects/easydeploy/tools/export.py \
	configs/yolox/yolox_s-v61_syncbn_fast_8xb16-300e_coco.py \
	yoloxs.pth \
	--work-dir work_dir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 1 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```


### step.2 准备数据集
- 准备[COCO](https://cocodataset.org/#download)数据集


### step.3 模型转换

1. 获取vamc模型转换工具

2. 根据具体模型修改模型转换配置文件[mmyolo_yolox.yaml](../vacc_code/build/mmyolo_yolox.yaml)：
    ```bash
    vamc build ../vacc_code/build/mmyolo_yolox.yaml
    ```

### step.4 性能精度
1. 获取vamp性能测试工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py --dataset_path path/to/coco_val2017 --target_path  path/to/coco_val2017_npz  --text_path npz_datalist.txt
    ```
3. 性能测试
    > vdsp_params中的color_cvt_code需要设置为COLOR_BGR2RGB，COLOR_NO_CHANGE会掉三到四个点
    ```bash
    vamp -m deploy_weights/yolox_s-int8-kl_divergence-3_640_640-vacc/yolox_s --vdsp_params ../vacc_code/vdsp_params/mmyolo-yolox_s-vdsp_params.json -i 2 p 2 -b 1
    ```
4. npz结果输出
    ```bash
    vamp -m deploy_weights/yolox_s-int8-kl_divergence-3_640_640-vacc/yolox_s --vdsp_params ../vacc_code/vdsp_params/mmyolo-yolox_s-vdsp_params.json -i 2 p 2 -b 1 --datalist datasets/coco_npz_datalist.txt --path_output npz_output
    ```
5. [vamp_decode_mmyolo.py](../vacc_code/vdsp_params/vamp_decode_mmyolo.py)，进行后处理，解析vamp输出的npz文件，进行绘图和保存txt结果
    > yolox_mmyolo未适配后处理算子，需要使用带后处理的npz解析脚本
    ```bash
    python ../vacc_code/vdsp_params/vamp_decode_mmyolo.py --input_image_dir datasets/coco_val2017 --vamp_datalist_path datasets/coco_npz_datalist.txt --vamp_output_dir npz_output --model_size 640 640 --save_dir output
    ```
6. [eval_map.py](../../common/eval/eval_map.py)，精度统计，指定`instances_val2017.json`标签文件和上步骤中的txt保存路径，即可获得mAP评估指标
   ```bash
    python ../../common/eval/eval_map.py --gt path/to/instances_val2017.json --txt path/to/vamp_draw_output
   ```

