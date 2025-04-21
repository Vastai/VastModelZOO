from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():

    from  opencompass.configs.datasets.gsm8k.gsm8k_gen import \
        gsm8k_datasets
    from  opencompass.configs.datasets.mmlu.mmlu_gen import \
        mmlu_datasets
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets  # 541sample
    from opencompass.configs.datasets.aime2024.aime2024_gen import aime2024_datasets # 问题少30个，问题短，回答长
    from opencompass.configs.datasets.FewCLUE_cluewsc.FewCLUE_cluewsc_gen import cluewsc_datasets # 159+976个 # 中文
    from opencompass.configs.datasets.math.math_500_gen import math_datasets  # 512个，问题/回答，长度适中
    from opencompass.configs.datasets.gpqa.gpqa_gen import gpqa_datasets # good 197个


# set dataset
datasets = cluewsc_datasets + math_datasets + mmlu_datasets

# api_meta_template = dict(round=[
#     dict(role='HUMAN', api_role='HUMAN'),
#     dict(role='BOT', api_role='BOT', generate=True),
# ], )

# api_meta_template = dict(
#     round=[
#         dict(role='HUMAN', api_role='HUMAN'),
#         dict(role='BOT', api_role='BOT', generate=True),
#     ],
#     reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
# )



######################################################### DeepSeek-V3 infer api ####################################################################
models = [
    dict(
        abbr='DeepSeek-V3-va16',
        type=OpenAISDK,
        path='/test/weights/DeepSeek-V3',
        openai_api_base=['http://10.24.73.25:8000/v1'],
        tokenizer_path='/test/weights/DeepSeek-V3',
        key=
        'token-abc123',
        # meta_template=api_meta_template,
        # query_per_second=2,
        max_out_len=8192,
        max_seq_len=16384,
        # temperature=0.0,
        batch_size=1),
]


# models = [
#     dict(
#         abbr='deepseek-v3-h800',
#         type=OpenAISDK,
#         path='/root/.cache/huggingface',
#         openai_api_base=['http://10.24.9.4:8000/v1'],
#         tokenizer_path='/root/.cache/huggingface',
#         key=
#         'token-abc123',
#         # meta_template=api_meta_template,
#         # query_per_second=2,
#         max_out_len=8192,
#         max_seq_len=16384,
#         # temperature=0.0,
#         batch_size=1),
# ]


###############################################################################################################################################



######################################################### DeepSeek-R1 infer api ####################################################################
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

# models = [
#     dict(
#         abbr='deepseek-r1-va16',
#         type=OpenAISDK,
#         path='/test/weights/DeepSeek-R1',
#         openai_api_base=['http://10.24.73.25:8000/v1'],
#         tokenizer_path='/test/weights/DeepSeek-R1',
#         key=
#         'token-abc123',
#         # meta_template=api_meta_template,
#         # query_per_second=2,
#         max_out_len=16384,
#         max_seq_len=32768, # 32768
#         temperature=0.6,
#         batch_size=1,
#         pred_postprocessor=dict(type=extract_non_reasoning_content)  # important to opencompass extract model output text
#         ),
# ]

# models = [
#     dict(
#         abbr='deepseek-r1-h800',
#         type=OpenAISDK,
#         path='/root/.cache/huggingface',
#         openai_api_base=['http://10.24.9.4:8000/v1'],
#         tokenizer_path='/root/.cache/huggingface',
#         key=
#         'token-abc123',
#         # meta_template=api_meta_template,
#         # query_per_second=2,
#         max_out_len=16384,
#         max_seq_len=32768, # 32768
#         temperature=0.6,
#         batch_size=1,
#         pred_postprocessor=dict(type=extract_non_reasoning_content)  # important to opencompass extract model output text
#         ),
# ]
###############################################################################################################################################