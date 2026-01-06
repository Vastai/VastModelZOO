# LLM Models

对大模型编译及性能测试相关的经验知识总结。

## COMMON
- 对于某一模型，如`Base`支持，`Chat`或`Instruct`也支持
- `Chat`或`Instruct`模型，推理测试时注意参考模型原始来源，设置正确的`prompt`
- `iter0`即`prefill`预填充阶段；`iter1`即`decoding`解码阶段

## VAMC模型编译
- 除LLaMa模型外，其它基于`Build-In`编译流程的模型，均需在原始模型基础上，进行以下操作：
    - 拷贝原始`config.json`文件，命名为`config_vacc.json`
        - 形如：[config_vacc.json](./qwen2/build_in/source_code/config_vacc.json)
        - 在`config_vacc.json`内增加`auto_map`字段，修改模型结构调用方式，从原始transformers内读取，修改为从新增`xxx_modeling_xxx.py`文件内读取
            - 此`xxx_modeling.py`文件在每个模型部署文档内提供（为支持对应模型在VACC硬件下推理，需做适当修改）
        - 新增参数项：`"_attn_implementation": "eager", "insert_slice": true`
    - 添加文档中描述的`xxx_modeling_xxx.py`等相关文件至模型文件夹内

- 模型编译序列长度**必须为16的倍数**
- 编译32b以上模型时需要添加环境变量, `export VACC_STACK_SIZE=256`
- 模型编译yaml表`frontend-model_kwargs`参数`b2s: true`开启可提升模型吞吐，建议性能测试中：
    - 测吞吐开启
    - 测时延关闭
- 模型编译yaml表`backend-compile`内设置参数`gather_data_vccl_dsp_enable: true`，可优化编译内存

- GQA模型中，设置模型切分`tp`数值，在一些情况下需要开启编译参数`${frontend.model_kwargs.align_qkv: true}`
    - 如模型参数`num_key_value_heads`能被`tp`整除，可关闭`align_qkv`
