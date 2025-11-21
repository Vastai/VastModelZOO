# Qwen2_VL

- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/pdf/2409.12191)
- https://hf-mirror.com/Qwen/Qwen2-VL-7B-Instruct

## Model Arch

![](../../images/vlm/qwen2_vl/arch.png)

### pre-processing

#### text encoder
- text encoder的预处理仅需要经过tokenizer转为相应token序列(预插入了image占位符)

#### image encoder
- 传统预处理包括：to_rgb -> to_array -> resize -> rescale -> normalize
- 以及后续特殊预处理（经VDSP自定义算子实现）：tile -> reshape -> transpose -> reshape
- 最后由image patches经patch_embed后进入VIT输出image_embeds

### post-processing
- llm decoder

### backbone
- `Qwen2_VL_7B`由`ViT-600M`和`Qwen2-7B`构成，通过`MLP`将视觉模型和语言模型进行对齐


### common
- 朴素动态分辨率：通过如下方式获取最佳缩放尺寸

    ```python
    def smart_resize(
        height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
    ):
        """Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.

        """
        if height < factor or width < factor:
            raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
        elif max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar
    ```

- 多模态旋转位置嵌入 (M-RoPE)
为了更有效地处理多模态输入（文本、图像和视频）中的位置信息，M-RoPE 基本原理是将传统的 RoPE 分解为时间、宽度和高度，同时捕捉不同模态的时空信息。
文本输入：三部分为相同的数字，与普通 RoPE 相同
图像输入：固定t，h和w根据 token 位置定义
视频输入：视频被视为帧序列，每一帧的t递增，h和w与图像输入相同
当模型输入包含多种模态时，每种模态的位置编号从前一个模态的最大位置 ID 加一开始。

![](../../images/vlm/qwen2_vl/mrope.png)

- 统一的图像和视频理解
Qwen2-VL 采用了一种混合训练机制，结合了图像和视频数据，确保其在图像理解和视频理解方面熟练掌握。 为了尽可能完整地保留视频信息，我们以每秒2帧的速率对每个视频进行采样。 此外，集成了深度为2的3D卷积来处理视频输入，使模型能够处理 3D 管道而不是 2D 补丁，从而使其能够处理更多视频帧而不增加序列长度。



### train
- 延续Qwen-VL，Qwen2-VL也采用了3-stage的训练过程：ViT训练 -> 全参数训练 -> LLM指令微调
- 多样性的数据包括：图像文本对、OCR 数据、图像文本的文章、VQA 数据、视频对话以及图像知识。来源于网站、开源数据集以及人造数据
- Qwen2-VL的LLM组件使用来自Qwen2的参数进行初始化；而 Qwen2-VL的视觉编码器使用来自DFN的ViT进行初始化，原始ViT中的固定位置嵌入被替换为RoPE-2D

## Build_In Deploy
- [deploy.md](./source_code/deploy.md)
- [deploy_gptq_int4.md](./source_code/deploy_gptq_int4.md)
