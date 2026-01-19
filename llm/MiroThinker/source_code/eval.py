#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from evalscope import TaskConfig, run_task
from evalscope.constants import JudgeStrategy


def run_frames_evaluation():
    generation_config = {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16384,
        "repetition_penalty": 1.05,
        "n": 1,
        "stream": True,
        "timeout": 6000000,
    }

    # LLM AS judge
    judge_model_args = {
        "model_id": "judge_model",
        "api_url": "http://10.24.9.2:8000/v1",
        "api_key": "abc",
        "generation_config": {"temperature": 0.0, "max_tokens": 4096},
        "score_type": "pattern",
    }

    task_cfg = TaskConfig(
        model="eval_model",
        api_url="http://0.0.0.0:9000/v1",
        api_key="abc",
        eval_type="openai_api",
        datasets=["frames"],
        repeats=3,
        generation_config=generation_config,
        eval_batch_size=32,
        judge_model_args=judge_model_args,
        judge_worker_num=5,
        judge_strategy=JudgeStrategy.LLM,
    )
    return run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    run_frames_evaluation()
