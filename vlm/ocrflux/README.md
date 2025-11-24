# OCRFlux

- [OCRFlux: Mastering Complex Layouts and Seamless Page Merging](https://ocrflux.pdfparser.io/#/blog)
- [huggingface](https://huggingface.co/ChatDOC/OCRFlux-3B)
- [demo](https://ocrflux.pdfparser.io/#/)
- [github](https://github.com/chatdoc-com/OCRFlux)

## Model Arch

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
- `OCRFlux-3B`是基于`Qwen2.5-VL-3B-Instruct`进行微调，使用了私有文档数据集以及olmOCR-mix-0225数据集的部分数据，提升了文档解析能力

## output
- 模型可以输出图片、表格bbox，但图片检出能力较差
- 跨页面内容，通过模型推理进行合并
- 官方推理：[inference.py](https://github.com/chatdoc-com/OCRFlux/blob/main/ocrflux/inference.py#L205)

## train
- 单页解析训练
   - 为确保 OCRFlux-3B 模型的高单页解析质量，我们使用私有的文档数据集来训练该模型。这些数据集包含大约 110 万页，主要来自金融和学术文档。所有文档都经过人工多轮检查标注，以确保高标注质量。此外，我们还使用了公共数据集 olmOCR-mix-0225 的部分数据（约 25 万页）。具体来说，我们发现 GPT-4o 提供的真实标签在包含表格的页面上质量较差，因此我们将其全部过滤掉用于训练。
   - 与之前的 olmOCR 等模型不同，我们的模型仅使用页面图像作为输入，而不使用任何元数据，例如文本块及其位置。这一决定与 RolmOCR 中的决定一致。这不会损害我们模型的准确性，并将显著减少提示长度，从而降低处理时间和内存消耗。此外，它还可以避免因损坏的元数据或 OCR 结果（如误读字符、阅读顺序错误和内容缺失）而可能产生的错误。
   - 由于 Markdown 无法自然地表示具有合并行 span 和列 span 的复杂表格，我们在训练数据中使用 HTML 格式来表示表格。表格解析的示例可以在交互式比较部分找到。

- 跨页段落/表格合并的训练
   - PDF 文档通常是分页的，这常常导致表格或段落被分割到连续的页面上。准确检测并合并这种跨页结构对于避免生成不完整或碎片化的内容至关重要。
   - 检测任务可以表述如下：给定两个连续页面的 Markdown——每个页面都结构化为 Markdown 元素的列表（例如段落和表格）——目标是识别应该跨页合并的元素的索引。
   - 然后对于合并任务，如果待合并的元素是段落，我们只需将它们连接起来。然而，对于两个表格片段，它们的合并要复杂得多。请参考交互式比较部分以获取详细示例。
   - 为了训练我们的模型进行检测和合并任务，我们在训练中使用了大约 450,000 个样本进行检测任务，以及 100,000 个样本进行合并任务。它们都来自我们的私有数据集。
   - 我们没有分别训练单页解析和跨页合并任务，而是在同一个多模态 LLM 中使用不同的提示将它们一起训练。这有助于将这两种能力集成到一个模型中，该模型在推理时可以更强大和高效。

## Build_In Deploy
- [deploy.md](./source_code/deploy.md)