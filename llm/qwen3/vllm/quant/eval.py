#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from evalscope import TaskConfig, run_task


def run_base_evaluation(model, generation_config):
    """运行基础评估任务"""
    task_cfg = TaskConfig(
        model=model,
        api_url='http://0.0.0.0:8012/v1/chat/completions',
        api_key="EMPTY",
        eval_type='service',
        datasets=['aime25', 'gpqa', 'live_code_bench', 'ifeval'],
        dataset_args={"gpqa": {"subset_list": ["gpqa_diamond"]}},
        eval_batch_size=32,
        generation_config=generation_config,
        stream=True,
        timeout=6000000,
    )
    return run_task(task_cfg=task_cfg)


def run_mmlu_pro_evaluation(model, generation_config):
    """运行MMLU Pro评估任务"""
    task_cfg = TaskConfig(
        model=model,
        api_url='http://0.0.0.0:8012/v1/chat/completions',
        api_key="EMPTY",
        eval_type='service',
        datasets=['mmlu_pro'],
        eval_batch_size=32,
        generation_config=generation_config,
        limit=0.1,
        stream=True,
        timeout=6000000,
    )
    return run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run model evaluations with shared configuration.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3-30B-A3B-Instruct-2507-gptq-w4a16-g128",
        help="Model name.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling.")
    parser.add_argument(
        "--max_tokens", type=int, default=16384, help="Maximum number of output tokens."
    )
    parser.add_argument("--n", type=int, default=1, help="Number of generations per input.")
    args = parser.parse_args()

    # 共享 generation_config
    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "n": args.n,
    }

    # 运行两个评估任务
    run_base_evaluation(args.model, generation_config)
    run_mmlu_pro_evaluation(args.model, generation_config)
