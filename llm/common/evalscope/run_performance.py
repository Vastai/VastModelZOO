import csv
import os
import math
from typing import List, Dict, Any
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.arguments import Arguments

from utils import save_to_csv

# 常量定义
MILLISECONDS_PER_SECOND = 1000

# 配置参数 - 建议改为从环境变量或配置文件中读取
model_name = "DeepSeek-R1-0528"
tokenizer_path = f"deepseek-ai/{model_name}"
url = os.getenv("API_URL", "http://127.0.0.1:8000/v1/chat/completions")
api = "openai"
api_key = os.getenv("API_KEY", "token-abc123")  # 从环境变量读取API密钥
dataset = "random"
max_concurrencies = [1, 2, 4]
pre_req_nums = 1
input_tokens_list = [128, 256, 512, 1024, 2048, 4096, 7900, 16384, 32768, 54272]
output_tokens_list = [1024, 8192]
max_seqlen = 65536
min_block_size = 8192
max_per_request = 57344
csv_file_path = "evalscope_benchmark.csv"

def process_results(results: List[Dict[str, Any]], concurrency: int, input_tokens: int) -> List[Dict[str, Any]]:
    """处理性能测试结果并返回格式化数据"""
    processed_data = []
    for result in results:
        ttft_str = result.get("Average time to first token (s)", "")
        tpot_str = result.get("Average time per output token (s)", "")
        
        try:
            ttft = float(ttft_str) if ttft_str else 0.0
            tpot = float(tpot_str) if tpot_str else 0.0
        except ValueError:
            ttft = 0.0
            tpot = 0.0
            
        if ttft <= 0.0 or tpot <= 0.0:
            continue
            
        decode_token_thr = MILLISECONDS_PER_SECOND / (tpot * MILLISECONDS_PER_SECOND)
        
        processed_data.append({
            "Maximum req": concurrency,
            "Duration (s)": result.get("Time taken for tests (s)", ""),
            "Successful req": result.get("Succeed requests", ""),
            "input tokens": input_tokens,
            "generated tokens": result.get("Average output tokens per request", ""),
            "Req throughput (req/s)": result.get("Request throughput (req/s)", ""),
            "Output token throughput (tok/s)": result.get("Output token throughput (tok/s)", ""),
            "Total Token throughput (tok/s)": result.get("Total token throughput (tok/s)", ""),
            "Mean TTFT (ms)": round(ttft * MILLISECONDS_PER_SECOND, 2),
            "Mean TPOT (ms)": round(tpot * MILLISECONDS_PER_SECOND, 2),
            "Decode Token throughput (tok/s)": round((decode_token_thr * concurrency), 2),
            "Per-req Decoding token throughput (tok/s)": round(decode_token_thr, 2),
            "model_name": model_name,
        })
    return processed_data

# 遍历参数组合并执行测试
for output_len in output_tokens_list:
    for concurrency in max_concurrencies:
        max_total_per_concurrency = max_seqlen // concurrency
        max_input_len = max_per_request
        
        if max_total_per_concurrency < output_len:
            print(f"[SKIP] Concurrency={concurrency}, output={output_len}: 总资源分配{max_total_per_concurrency} < 输出长度{output_len}")
            continue

        valid_inputs = [
            input_len for input_len in input_tokens_list 
            if math.ceil(input_len/min_block_size)*min_block_size <= max_input_len and math.ceil((input_len + output_len)/min_block_size)*min_block_size <= max_total_per_concurrency
        ]

        if not valid_inputs:
            print(f"[SKIP] Concurrency={concurrency}, output={output_len}: 无有效输入长度（最大允许input_len={max_input_len}, 每用户总长度限制={max_total_per_concurrency}）")
            continue

        print(f"Concurrency={concurrency}, output={output_len}, 有效输入长度: {valid_inputs}")
        
        for input_tokens in valid_inputs:  # 只遍历有效输入长度
            print(f"Concurrency={concurrency}, output={output_len}, input={input_tokens}")
            
            task_cfg = Arguments(
                parallel=[concurrency],
                number=[pre_req_nums * concurrency],
                model=model_name,
                url=url,
                api=api,
                api_key=api_key,
                dataset=dataset,
                read_timeout=60000,
                min_tokens=output_len,
                max_tokens=output_len,
                prefix_length=0,
                min_prompt_length=input_tokens,
                max_prompt_length=input_tokens,
                tokenizer_path=tokenizer_path,
                extra_args={'ignore_eos': True},
                #no_test_connection=True,
            )

            try:
                results = run_perf_benchmark(task_cfg)
                print(f"results:{results}")
                
                if results:
                    data = process_results(results, concurrency, input_tokens)
                    if data:
                        save_to_csv(data, csv_file_path)
            except Exception as e:
                print(f"Error running benchmark for concurrency={concurrency}, output={output_len}, input={input_tokens}: {str(e)}")
                continue

print(f"Performance test results have been saved to {csv_file_path}")