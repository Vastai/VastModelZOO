
# GPEN

[GAN Prior Embedded Network for Blind Face Restoration in the Wild](https://arxiv.org/abs/2105.06070)

## Code Source
```
link: https://github.com/yangxy/GPEN
branch: master
commit: b611a9f2d05fdc5b82715219ed0f252e42e4d8be
```

## Model Arch

<div align=center><img src="../../../images/cv/super_resolution/gpen/arch.png"></div>

### pre-processing

GPEN系列网络的预处理操作，可以按照如下步骤进行：

```python
  process_ops:
    - type: DecodeImage
    - type: Resize
      size: [512, 512]
    - type: Normalize
      mean: [0.5, 0.5, 0.5]
      std: [0.5, 0.5, 0.5]
    - type: ToTensor
```

### post-processing

GPEN系列网络的后处理操作，可以按照如下步骤进行：
```python
output = np.squeeze(heatmap)
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
output = output * 0.5 + 0.5

output = np.clip(output, 0, 1) * 255.0
output = output.astype(np.uint8)
```

### backbone

从野外严重退化的人脸图像中恢复盲脸（BFR）是一个非常具有挑战性的问题。由于较大的畸变和退化，DNN难以在盲脸恢复中取得较好效果，现有GAN偏向生成过于平滑的修复结果。

作者先采用高精度人脸数据训练出GAN（StyleGAN2）生成器，将其作为先验解码器嵌入到U形DNN网络中，对输入低质量图像进行微调；
GAN 模块旨在确保输入到 GAN 的潜在代码和噪声可以分别从 DNN 的深层和浅层特征生成，控制重建图像的全局人脸结构、局部人脸细节和背景。所提出的 GAN 先验嵌入式网络 (GPEN) 易于实现，并且可以生成视觉上逼真的结果。

<div align=center><img src="../../../images/cv/super_resolution/gpen/gpen.png"></div>

图(a)是GAN先验网络，受StyleGAN结构的启发，使用映射网络将潜在代码z经过一个映射网络投射到空间w中。然后将中间代码w传播到每个GAN块。每个GAN采用的StyleGAN块体系结构，如图(b)
由于GAN-prior网络将嵌入到U shaped-DNN中进行微调，因此我们需要为Ushaped-DNN编码器提取的特征映射留出空间。因此，我们为每个GAN块提供额外的噪声输入。
此外，噪声输入是串联的，而不是添加到StyleGAN中的卷积中。这个串联方法可以在恢复的人脸图像中带来更多细节，作者也在后面的消融实验中进行了证明。
当使用某个数据集训练了GAN先验网络，我们将其作为解码器嵌入U形DNN中，如图(c)。潜在代码z被DNN编码器的全连接层所代替， GAN网络的噪声输入被编码器每一层的输出所代替，这将控制全局人脸结构、局部人脸细节以及人脸图像背景的重建


### common

- U shaped-DNN
- StyleGAN

## Model Info

### 模型性能

| Models  | Flops(G) | Params(M) | PSNR(dB) | SSIM | Shapes |
| :---: | :--: | :--: | :---: | :----: | :--------: |
| [GPEN](https://github.com/yangxy/GPEN) |  -  |  71.010  | 25.036  | 0.936 |  3x512x512  |
| GPEN **vacc fp16** |  -  |  -  |  - | - |  -  |
| GPEN **vacc mse int8** |  -  |  -  |  22.246 |  0.821 |  3x512x512  |


### 测评数据集说明

[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)是一个大规模的面部图像数据集，通过遵循CelebA-HQ从CelebA数据集中选择了30,000张高分辨率面部图像。 每个图像具有对应于CelebA的面部属性的分割MASK，其采用512 x 512尺寸手动标注，分为19类，包括所有面部组件和配件，例如皮肤，鼻子，眼睛，眉毛，耳朵，嘴巴，嘴唇，头发，帽子，眼镜，耳环，项链，脖子和布。CelebAMask-HQ可用于训练和评估人脸解析，人脸识别以及用于人脸生成和编辑的GAN的算法。

<div  align="center">
<img src="../../../images/dataset/celebamask-hq.png" width="60%" height="60%">
</div>

### 指标说明
- 峰值信噪比(Peak Signal-to-Noise Ratio, PSNR)，PSNR是信号的最大功率和信号噪声功率之比，测量重构图像的质量，通常以分贝（dB）来表示。PSNR指标越高，说明图像质量越好
- 结构相似性评价(Structure Similarity Index, SSIM)，SSIM是衡量两幅图像相似度的指标，其取值范围为[0,1]，SSIM的值越大，表示图像失真程度越小，说明图像质量越好
- Fréchet Inception Distance，FID是衡量两个多元正态分布的距离，反映了生成图片和真实图片的距离，数据越小越好


## Build_In Deploy

### step.1 获取预训练模型
- 拉取代码至`source_code`目录下
- 将[export.py](./source_code/export.py)移动至`source_code/GPEN`目录下
- 修改[gpen_model.py#L690](https://github.com/yangxy/GPEN/blob/main/face_model/gpen_model.py#L690)，在return前添加以下代码：
    ```python
    if len(outs) == 2:
        if outs[1] is None:
            outs = outs[0]

    return outs
    ```
- 执行转换脚本，得到`onnx`和`torchscript`：
    ```python
    python super_resolution/gpen/source_code/GPEN/export.py
    ```
#### Tips
- GPEN默认会将灰度图人脸恢复成彩色图人脸，尺寸没变化

### step.2 准备数据集
- 按论文，取[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)数据集的前1000张图像作为HQ
- 基于[hq2gray.py](./source_code/hq2gray.py)，使用将彩色高清图像转换为灰度图像
- 基于[image2npz.py](../common/utils/image2npz.py)，将灰度图像转为npz格式

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_gpen.yaml](./build_in/build/official_gpen.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd gpen
    mkdir workspace
    cd workspace
    vamc compile ../build_in/build/official_gpen.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考：[official_vsx_inference.py](./build_in/vsx/python/official_vsx_inference.py)
    ```bash
    python ../build_in/vsx/python/official_vsx_inference.py \
        --lr_image_dir  /path/to/GPEN/hq_gray/ \
        --model_prefix_path deploy_weights/official_gpen_run_stream_fp16/mod \
        --vdsp_params_info ../build_in/vdsp_params/official-gpen-vdsp_params.json \
        --hr_image_dir /path/to/GPEN/hq \
        --save_dir ./runstream_output \
        --device 0
    ```

    ```
    # fp16
    mean psnr: 21.95145274270508, mean ssim: 0.8669498382798001

    # int8
    mean psnr: 22.29287021952575, mean ssim: 0.8196632045742404
    ```

### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-gpen-vdsp_params.json](./build_in/vdsp_params/official-gpen-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/official_gpen_run_stream_int8/mod \
    --vdsp_params ../build_in/vdsp_params/official-gpen-vdsp_params.json \
    -i 1 p 1 -b 1 -s [3,512,512]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`datalist_npz_gray.txt`：
    ```bash
    python ../common/utils/image2npz.py \
        --dataset_path GPEN/hq_gray \
        --target_path GPEN/hq_gray_npz \
        --text_path datalist_npz_gray.txt
    ```

    - vamp推理得到npz结果：
    ```bash
    vamp -m deploy_weights/official_gpen_run_stream_int8/mod \
        --vdsp_params ../build_in/vdsp_params/official-gpen-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,512,512] \
        --datalist datalist_npz_gray.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[vamp_eval.py](./build_in/vdsp_params/vamp_eval.py)，
    ```bash
    python ./build_in/vdsp_params/vamp_eval.py \
        --gt_dir GPEN/hq \
        --input_npz_path datalist_npz_gray.txt \
        --out_npz_dir npz_output \
        --input_shape 512 512 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```

## Tips
- 当前只支持torchscript模型编译
- GPEN有多个模型，可实现人脸恢复、人脸上色等任务
- 仓库提供的face colorization模型为1024尺寸；此处实现，基于FFHQ数据集自训练的512尺寸，`facegan = FaceGAN(base_dir=".", in_size=512, out_size=512, model='GPEN', channel_multiplier=2, narrow=1,  device="cpu")`
- build参数和原GPEN(facesr)一致
