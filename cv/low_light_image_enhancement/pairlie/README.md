
# PairLIE

[Learning a Simple Low-light Image Enhancer from Paired Low-light Instances](https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.pdf)

## Code Source
```
# official
link: https://github.com/zhenqifu/PairLIE/
branch: main
commit: 3fd5c1586aa5f8f0736382de36d5bf46a4e208c2
```

## Model Arch

<div align=center><img src="../../../images/cv/low_light_image_enhancement/pairlie/arch.png"></div>

### pre-processing

PairLIE系列网络的预处理操作，可以按照如下步骤进行：

```python
def get_image_data(image_file, input_shape = [1, 3, 400, 600]):
    size = input_shape[2:][::-1]

    image = cv2.imread(image_file)
    img = cv2.resize(image, size) # , interpolation=cv2.INTER_AREA
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) # HWC to CHW
    img /= 255.0
    img = np.expand_dims(img, axis=0)

    return np.array(img)
```

### post-processing

PairLIE系列网络的后处理操作，可以按照如下步骤进行：

```python
heatmap = vacc_model.get_output(name, 0, 0).asnumpy().astype("float32")

output = np.squeeze(heatmap)
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
output = np.clip(output * 255, 0, 255)
```

### backbone

利用成对的弱光实例，文中提出了一种新的基于学习的LIE方法，称为PairLIE，它的核心见解是充分利用来自成对的低光图像的先验。因此，考虑使用Retinex理论和深度学习将弱光图像分解为照明和反射分量。首先，因为两个弱光输入共享同样的内容，所以估计的反射率分量预计是一致的。其次，放弃直接在原低光图像中进行Retinex分解，本文采用一个简单的自监督机制去除不合适的特征，对优化后的图像进行Retinex分解。这可以避免次优估计，因为Retinex模型在弱光建模中有局限性。在较少的先验约束和更简单的网络下，所提出的PairLIE在公共LIE数据集中实现了具有竞争力的性能。

### common

- ReLU
- ReflectionPad2d

## Model Info

### 模型性能

| Models  |  Code Source |Flops(G) | Params(M) | PSNR(dB) | SSIM | Shape |
| :---: | :--: |:--: | :--: | :---: | :----: | :--------: |
| PairLIE | [Official](https://github.com/zhenqifu/PairLIE/) |  182.276  |  0.342  |  18.498 | 0.748 | 3x400x600 |
| PairLIE **vacc fp16** |  -  |  -  |  -  |  18.288 |  0.745 |  3x400x600  |
| PairLIE **vacc kl_divergence int8** |  -  |  -  |  -  |  18.280 |  0.722 |  3x400x600  |

> Tips
> - 精度数值，基于LOL，15 eval图像


### 测评数据集说明

[LOL](https://daooshee.github.io/BMVC2018website/)，全称LOw-Light dataset，由500个弱光和正常光的图像对组成。
- 该数据集分为485个训练对和15个测试对
- 弱光图像包含在照片拍摄过程中产生的噪点
- 大多数图像是室内场景
- 所有图像的分辨率为 400×600 像素

<div  align="center">
<img src="../../../images/dataset/lol.png" width="70%" height="70%">
</div>



### 评价指标说明
- 峰值信噪比(Peak Signal-to-Noise Ratio, PSNR)，PSNR是信号的最大功率和信号噪声功率之比，测量重构图像的质量，通常以分贝（dB）来表示。PSNR指标越高，说明图像质量越好
- 结构相似性评价(Structure Similarity Index, SSIM)，SSIM是衡量两幅图像相似度的指标，其取值范围为[0,1]，SSIM的值越大，表示图像失真程度越小，说明图像质量越好
- Fréchet Inception Distance，FID是衡量两个多元正态分布的距离，反映了生成图片和真实图片的距离，数据越小越好


## Build_In Deploy

### step.1 获取预训练模型

- 原始模型在forward时返回三个参数，只需要前两个，修改[net/net.py#L88](https://github.com/zhenqifu/PairLIE/blob/main/net/net.py#L88)，为`return L, R`
- 参考[export.py](./source_code/official/export.py)，将此文件移动至原始仓库，并修改原始权重路径，导出torchscript和onnx

### step.2 准备数据集
- 下载[LOL](https://daooshee.github.io/BMVC2018website/)数据集
- 通过[image2npz.py](../common/utils/image2npz.py)，将测试低照度LR图像`eval15`转换为对应npz文件

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_pairlie.yaml](./build_in/build/official_pairlie.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd pairlie
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_pairlie.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考：[official_vsx_inference.py](./build_in/vsx/python/official_vsx_inference.py)
    ```bash
    python ./build_in/vsx/python/official_vsx_inference.py \
        --lr_image_dir  /path/to/lol/eval15/low/ \
        --hr_image_dir /path/to/lol/eval15/high \
        --model_prefix_path deploy_weights/official_pairlie_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-pairlie-vdsp_params.json \
        --save_dir ./runstream_output \
        --device 0
    ```

    ```
    # fp16
    mean psnr: 18.49810352310849, mean ssim: 0.7477113710201562

    # int8
    mean psnr: 18.401180004200523, mean ssim: 0.726852211962538
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-pairlie-vdsp_params.json](./build_in/vdsp_params/official-pairlie-vdsp_params.json)
    ```bash
    vamp -m deploy_weights/official_pairlie_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-pairlie-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,400,600]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../common/utils/image2npz.py \
        --dataset_path low_light_enhance/LOL/eval15/low \
        --target_path  low_light_enhance/LOL/eval15/low_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果
    ```bash
    vamp -m deploy_weights/official_pairlie_run_stream_int8/mod \
        --vdsp_params ./build_in/vdsp_params/official-pairlie-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,400,600] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考[official-vamp_eval.py](./build_in/vdsp_params/official-vamp_eval.py)
   ```bash
    python ../build_in/vdsp_params/official-vamp_eval.py \
        --gt_dir low_light_enhance/lol/eval15/high \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir npz_output/ \
        --input_shape 400 600 \
        --draw_dir npz_draw_result \
        --vamp_flag
   ```
