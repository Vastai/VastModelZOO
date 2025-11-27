# CLIP

[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020)

## Model Arch

### pre-processing
#### text encoder
text encoder的预处理仅需要经过tokenizer转为相应token序列
#### image encoder
image encoder的预处理如下所示
```python
def _convert_image_to_rgb(image):
    return image.convert("RGB")

Compose([
    Resize(n_px, interpolation=BICUBIC),
    CenterCrop(n_px),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
```
### post-processing
clip的后处理是将两个编码器得到的同维度图像嵌入和文本嵌入， 通过矩阵相乘映射到相同的向量空间， 计算图像-文本对的相似度代码如下
```python
def forward(self, image, text):
    # encoder
    image_features = self.encode_image(image)
    text_features = self.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = self.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text
```
### backbone
text encoder采用的是gpt2风格transformer encoder模型， image encoder 采用的是resnet（或者vit）
### common
- Constastive Pretraining

## Build_In Deploy
- [openai_deploy.md](./source_code/openai_deploy.md)