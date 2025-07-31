## VDSR

[Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf)


## Code Source
```
link: https://github.com/twtygqyy/pytorch-vdsr
branch: main
commit: 514b021044018baf909e79f48392783daa592888
```

## Model Arch

<div  align="center">
<img src="../../../images/cv/super_resolution/vdsr/vdsr.png" width="70%" height="70%">
</div>

### pre-processing

VDSR网络的预处理操作可以按照如下步骤进行，即对图片进行resize至一定尺寸(256)，然后除以255：

```python
def get_image_data(image_file, input_shape = [1, 1, 256, 256]):
    """fix shape resize"""
    size = input_shape[2:]

    im_b_ycbcr = cv2.imread(image_file)
    im_b_ycbcr = cv2.cvtColor(im_b_ycbcr, cv2.COLOR_BGR2YCrCb)
    im_b_ycbcr = cv2.resize(im_b_ycbcr, size)
    
    im_b_y = im_b_ycbcr[:, :, 0].astype(float)
    im_input = im_b_y / 255.

    img_data = im_input[np.newaxis, np.newaxis, :, :] # NCHW

    return im_b_ycbcr, img_data

```
> Tips
> 
> 模型训练时输入为YCbCr颜色空间的Y通道，input shape (1, 1, 256, 256)


### post-processing

VDSR模型的后处理，对模型输出乘以255，像素恢复至[0, 255]，然后再添加回Cb、Cr颜色分量，得到最终高分辨率图像。

### backbone

VDSR模型使用了vgg19作为骨架网络。

作者使用20个网络层，除第一层和最后一层外，其余层具有相同的类型：64个大小为3x3x64的滤波器。输入层是经插值的低分辨率(ILR)图像经过层层转换成高分辨率(HR)图像。网络预测残差图像，ILR和残差相加得到期望的输出（LR和HR图像很大程度上是相似的，它们的低频信息相近，所不同的是LR缺少了很多高频信息。即：输出=低分辨输入+学习到的残差）。

- VDSR是在SRCNN的基础上，加深了网络的深度，最终的网络模型达到20层，可实现多尺度的超分辨率生成
- 在深层网络结构中多次级联小型滤波器，可以有效的利用大型图像区域上的上下文信息
- 通过仅学习残差和使用极高的学习率来改善深层网络收敛慢的问题
- 在训练阶段，SRCNN直接对高分辨率图像建模。一个高分辨率图像能够被分解为低频信息（对应低分辨率图像）和高频信息（残差图像或图像细节）。而输入和输出图像共享相同的低频信息。这说明SRCNN有两个作用：携带输入（我理解就是携带与输出图像共享的相同低频信息到终端层）到终端层和重建残差图像（这样最后将可以输出高分辨率图像了）。训练时间可能会花费在学习“携带输入到终端层”这一过程上，这样会导致重建残差图像过程的收敛率大大下降。相比SRCNN，VDSR网络直接对残差图像进行建模，所以有更快的收敛速度，甚至更好的精度。在训练的过程中，VDSR通过填充0，使得输出图像具有和输入图像相同的尺寸大小，而SRCNN模型的输出图像的尺寸小于输入图像。并且VDSR对所有层使用相同的学习率，而SRCNN为了实现稳定的收敛，对不同层使用不同的学习率


### common

- DNN
- Residual Block


## Model Info

## 模型精度

| Model | flops(G)| Params(M)| PSNR| Size |
|:-:|:-:|:-:|:-:|:-:|
| [VDSR](https://github.com/hamidreza-dastmalchi/WIPA-Face-Super-Resolution) |  96.805 |  0.664 | 31.464 | multi-size |
| [VDSR](https://github.com/hamidreza-dastmalchi/WIPA-Face-Super-Resolution) |  96.805 |  0.664 | 30.958 |  1x256x256 |
| VDSR **vacc fp16**| - | - | 30.956 |  1x256x256 |
| VDSR **vacc int8 kl_divergence**| - |  - | 30.219 | 1x256x256 |


### 测评数据集说明

[Set5](https://github.com/twtygqyy/pytorch-vdsr/tree/master/Set5)是基于非负邻域嵌入的低复杂度单图像超分辨率的数据集（共5张BMP图像），该训练集被用于单幅图像超分辨率重构，即根据低分辨率图像重构出高分辨率图像以获取更多的细节信息。

<div  align="center">
<img src="../../../images/dataset/Set5.png" width="80%" height="80%">
</div>

### 指标说明
- 峰值信噪比(Peak Signal-to-Noise Ratio, PSNR)，PSNR是信号的最大功率和信号噪声功率之比，测量重构图像的质量，通常以分贝（dB）来表示。PSNR指标越高，说明图像质量越好
- 结构相似性评价(Structure Similarity Index, SSIM)，SSIM是衡量两幅图像相似度的指标，其取值范围为[0,1]，SSIM的值越大，表示图像失真程度越小，说明图像质量越好
- Fréchet Inception Distance，FID是衡量两个多元正态分布的距离，反映了生成图片和真实图片的距离，数据越小越好

## Build_In Deploy

### step.1 获取预训练模型
一般在原始仓库内进行模型转为onnx或torchscript。在原仓库test或val脚本内，如[eval.py#L34](https://github.com/twtygqyy/pytorch-vdsr/blob/master/eval.py#L34)，定义模型和加载训练权重后，添加以下脚本可实现：

```python
model.eval()

input_shape = (1, 1, 256, 256)
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
torch.jit.save(scripted_model, 'drrn.torchscript.pt')

import onnx
torch.onnx.export(model, input_data, 'drrn.onnx', input_names=["input"], output_names=["output"], opset_version=10,
            # dynamic_axes= {
            #                 "input": {0: 'batch_size', 2 : 'in_height', 3: 'in_width'},
            #                 "output": {0: 'batch_size', 2: 'out_height', 3:'out_width'}
            #                 }
)
```

### step.2 准备数据集
- 下载[Set5_BMP](https://github.com/twtygqyy/pytorch-vdsr/tree/master/Set5)数据集

### step.3 模型转换
1. 根据具体模型，修改编译配置
    - [official_vdsr.yaml](./build_in/build/official_vdsr.yaml)
    
    > - runstream推理，编译参数`backend.type: tvm_vacc`
    > - fp16精度: 编译参数`backend.dtype: fp16`
    > - int8精度: 编译参数`backend.dtype: int8`，需要配置量化数据集和预处理算子

2. 模型编译
    ```bash
    cd vdsr
    mkdir workspace
    cd workspace
    vamc compile ./build_in/build/official_vdsr.yaml
    ```

### step.4 模型推理
1. runstream
    - 参考[vsx_inference.py](./build_in/vsx/python/vsx_inference.py)
    ```bash
    python ./build_in/vsx/python/vsx_inference.py \
        --lr_image_dir  /path/to/Set5_BMP/scale_4 \
        --model_prefix_path deploy_weights/official_vdsr_run_stream_int8/mod \
        --vdsp_params_info ./build_in/vdsp_params/official-vdsr-vdsp_params.json \
        --hr_image_dir /path/to/Set5_BMP/hr \
        --save_dir ./runstream_output \
        --device 0
    ```
    
    ```
    # fp16
    mean psnr: 30.914584837892953

    # int8 
    mean psnr: 30.242492457515436
    ```


### step.5 性能精度测试
1. 性能测试
    - 配置vdsp参数[official-vdsr-vdsp_params.json](./build_in/vdsp_params/official-vdsr-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/official_vdsr_run_stream_int8/mod \
        --vdsp_params ./build_in/vdsp_params/official-vdsr-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,256,256]
    ```

2. 精度测试
    > **可选步骤**，通过vamp推理方式获得推理结果，然后解析及评估精度；与前文基于runstream脚本形式评估精度效果一致

    - 数据准备，基于[image2npz.py](./build_in/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`（注意只需要YCrcb颜色空间的Y通道信息）：
    ```bash
    python ./build_in/vdsp_params/image2npz.py \
        --dataset_path Set5_BMP/scale_4 \
        --target_path  Set5_BMP/scale_4_npz \
        --text_path npz_datalist.txt
    ```

    - vamp推理得到npz结果
    ```bash
    vamp -m deploy_weights/official_vdsr_run_stream_int8/mod \
        --vdsp_params ./build_in/vdsp_params/pytorch-vdsr-vdsp_params.json \
        -i 1 p 1 -b 1 -s [3,256,256] \
        --datalist npz_datalist.txt \
        --path_output npz_output
    ```

    - 解析npz结果并统计精度，参考：[vamp_eval.py](./build_in/vdsp_params/vamp_eval.py)，
    ```bash
    python ./build_in/vdsp_params/vamp_eval.py \
        --src_dir Set5_BMP/scale_4 \
        --gt_dir Set5_BMP/hr \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir outputs/drrn \
        --input_shape 256 256 \
        --draw_dir npz_draw_result \
        --vamp_flag
    ```

