#!/bin/bash

# 定义可配置参数

MODEL_NAME="Qwen2.5-7B-Instruct-int8-tp8-2048-4096"
TOKENIZER_PATH="/path/to/Qwen2.5-7B-Instruct-int8-tp8-2048-4096/tokenizer"
SERVER_PORT=9900

MAX_INPUT_LEN=2048
MAX_SEQLEN=4096

PER_REQ_NUM_PROMPTS=5
INPUT_LENS=(510 1020 2040)
OUTPUT_LENS=(512 1024 2048)
MAX_CONCURRENCIES=(1 4 8 16)


# 使用本地日期初始化日志目录
CURRENT_DATE=$(date +"%Y%m%d%H%M%S")
LOG_DIR="./benchmark_logs_${CURRENT_DATE}"
LOG_FILE="${LOG_DIR}/benchmark.log"

# 创建日志目录和文件
mkdir -p "$LOG_DIR" || { echo "无法创建日志目录 $LOG_DIR"; exit 1; }
> "$LOG_FILE" || { echo "无法清空日志文件 $LOG_FILE"; exit 1; }

# 定义带时间戳的日志函数
log() {
    local message="$1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $message" | tee -a "$LOG_FILE"
}

# 记录开始时间
start_time=$(date +%s)
log "=== Benchmark Started ==="
log "System Constraints:"
log "  MAX_SEQLEN  : $MAX_SEQLEN "
log "  MAX_INPUT_LEN    : $MAX_INPUT_LEN "
log "Testing Combinations:"
log "  Concurrency Levels : ${MAX_CONCURRENCIES[*]}"
log "  Output Lengths     : ${OUTPUT_LENS[*]}"
log "----------------------------------------"

# 遍历所有参数组合
for output_len in "${OUTPUT_LENS[@]}"; do
    for concurrency in "${MAX_CONCURRENCIES[@]}"; do
        for input_len in "${INPUT_LENS[@]}"; do
            if (( input_len > MAX_INPUT_LEN )); then 
                log "[SKIP] concurrency=$concurrency, input_len=$input_len 大于 MAX_INPUT_LEN=$MAX_INPUT_LEN"
                continue
            fi
            max_output_len=$(( MAX_SEQLEN - input_len ))
            if (( output_len > max_output_len )); then 
                log "[SKIP] concurrency=$concurrency, output_len=$output_len 大于 max_output_len=$max_output_len"
                continue
            fi
            
            log "[START] concurrency=$concurrency, input=$input_len, output=$output_len"

            # 执行基准测试（带时间戳的详细日志）
            timestamp=$(date +"%Y%m%d_%H%M%S")
            result_file="${LOG_DIR}/${timestamp}_c${concurrency}_i${input_len}o${output_len}.json"

            start_test=$(date +%s)
            
            has_error=0
            while read line; do
                echo "$line" | awk '{print strftime("[%Y-%m-%d %H:%M:%S]")" [PYTHON] "$0}' >> "$LOG_FILE"
                if [[ $has_error -ne 0 ]] || [[ "$line" == *"Traceback"* ]] || [[ "$line" == *"Exception"* ]]; then
                    echo "$line" 
                    has_error=1
                fi
            done < <(python3 benchmark_serving.py \
                        --host 127.0.0.1 \
                        --port "$SERVER_PORT" \
                        --model "$MODEL_NAME" \
                        --dataset-name "random" \
                        --endpoint "/v1/chat/completions" \
                        --backend "openai-chat" \
                        --tokenizer "$TOKENIZER_PATH" \
                        --num-prompts $(( PER_REQ_NUM_PROMPTS * concurrency )) \
                        --random-input-len $input_len \
                        --ignore-eos \
                        --random-output-len $output_len \
                        --max-concurrency $concurrency \
                        --save-result \
                        --result-filename "$result_file" 2>&1 )

            if [[ $has_error -ne 0 ]]; then
                log "[ERROR] 测试失败！concurrency=$concurrency,input=$input_len,output=$output_len"
                exit 1
            fi

            duration=$(( $(date +%s) - start_test ))

            log "[END] 耗时${duration}s | 资源使用：$(( concurrency * (input_len + output_len) ))/${MAX_SEQLEN}"
        done
    done
done

# 生成总结报告
total_duration=$(( $(date +%s) - start_time ))
log "============================================="
log "基准测试完成"
log "总耗时: $(( total_duration / 3600 ))小时 $(( (total_duration % 3600) / 60 ))分钟"
log "日志目录: $LOG_DIR"
log "关键参数:"
log "  MAX_SEQLEN     = $MAX_SEQLEN"
log "  MAX_INPUT_LEN  = $MAX_INPUT_LEN (仅限制输入长度)"
log "============================================="
