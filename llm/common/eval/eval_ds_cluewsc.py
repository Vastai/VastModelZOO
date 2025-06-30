from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType


task_cfg = TaskConfig(
    model='DeepSeek-R1-0528',   # 模型名称 (需要与部署时的模型名称一致)
    api_url='http://10.24.73.200:8000/v1',  # 推理服务地址
    api_key='EMPTY',
    eval_type=EvalType.SERVICE,   # 评测类型，SERVICE表示评测推理服务
    datasets=['general_mcq'],  # 数据格式，选择题格式固定为 'general_mcq'
    dataset_args={
        'general_mcq': {
            "local_path": "./datasets/cluewsc_custom",  # 自定义数据集路径
            "subset_list": [
                "cluewsc",  # 评测数据集名称，上述subset_name
            ],
            "prompt_template": "以下问题的答案有AB两个选项，选出正确答案，请直接回答A或B\n\n{query}",
            "eval_split": 'test' 

        }
    },
    eval_batch_size=16,       # 发送请求的并发数
    generation_config={       # 模型推理配置
        'max_tokens': 61440,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,   # 采样温度 (deepseek 报告推荐值)
        'top_p': 0.95,        # top-p采样 (deepseek 报告推荐值)
        'n': 1                # 每个请求产生的回复数量 (注意 lmdeploy 目前只支持 n=1)
    },
    stream=True,               # 是否使用流式请求，推荐设置为True防止请求超时
    timeout=6000000,
    limit=50
)
run_task(task_cfg=task_cfg)