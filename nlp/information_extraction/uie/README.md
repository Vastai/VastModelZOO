## UIE

[UIE：Universal Information Extraction](https://arxiv.org/pdf/2203.12277)

## Code_Source
```
link：https://github.com/HUSTAI/uie_pytorch
branch: main
commit: 2eafcda44589144d2cb246b74e3bf2564ea6583f
```


## Model Arch

通用信息抽取统一框架UIE，实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。

<div align="center">
    <img src=../../../images/information_extraction/uie/arch.png height=400 hspace='10'/>
</div>

### UIE的优势

- 使用简单：用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。实现开箱即用，并满足各类信息抽取需求。

- 降本增效：以往的信息抽取技术需要大量标注数据才能保证信息抽取的效果，为了提高开发过程中的开发效率，减少不必要的重复工作时间，开放域信息抽取可以实现零样本（zero-shot）或者少样本（few-shot）抽取，大幅度降低标注数据依赖，在降低成本的同时，还提升了效果。

- 效果领先：开放域信息抽取在多种场景，多种任务上，均有不俗的表现。


## VACC部署
- [hustai_uie.md](./source_code/hustai_uie.md)
