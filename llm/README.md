# LLM Models

对大模型编译及性能测试相关的经验知识总结。

## COMMON
- 对于某一模型，如`Base`支持，`Chat`或`Instruct`也支持
- `Chat`或`Instruct`模型，推理测试时注意参考模型原始来源，设置正确的`prompt`
- `iter0`即`prefill`预填充阶段；`iter1`即`decoding`解码阶段

## VAMC模型编译
- 模型编译序列长度**必须为16的倍数**
- 编译70b时需要添加`环境变量`, `export VACC_STACK_SIZE=256`
- 模型编译yaml表`frontend-model_kwargs`参数`b2s: true`开启可提升模型吞吐，建议性能测试中：
    - 测吞吐开启
    - 测时延关闭
- 模型编译yaml表`backend-compile`内设置参数`gather_data_vccl_dsp_enable: true`，可优化编译内存

- GQA模型中，设置模型切分`tp`数值，在一些情况下需要开启编译参数`${frontend.model_kwargs.align_qkv: true}`
    - 如模型参数`num_key_value_heads`能被`tp`整除，可关闭`align_qkv`

## LLMDeploy性能测试
- llmdeploy精度测试中
    - 预训练Base模型，采用`困惑度模式PPL`
    - 有监督微调Chat模型，采用`生成模式GEN`
    - 注意PPL模式，编译模型时iter0的seq_len长度，需大于测评数据每个输入样本token长度
