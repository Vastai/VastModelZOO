#!/bin/bash
# 获取service_num参数（默认为1）
SERVICE_NUM=${1:-1}
if ! [[ "$SERVICE_NUM" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：service_num 必须是正整数"
    exit 1
fi

# 计算并发级别数组
MAX_CONCURRENCIES=()
max_concurrency=$((4 * SERVICE_NUM))
for ((i=1; i<=max_concurrency; i*=2)); do
    MAX_CONCURRENCIES+=($i)
done

export OPENAI_API_KEY="token-abc123"
# 定义可配置参数
MODEL_PATH="/weights/DeepSeek-V3-0324"
MODEL_NAME="DeepSeek-V3-0324"
HOST="127.0.0.1"
PORT=8000
DATASET_NAME="random"
PER_REQ_NUM_PROMPTS=5
INPUT_LENS=(128 256 512 1024 2048 4096 7900 16384 32768 54272)
OUTPUT_LENS=(1024 8192)

MAX_SEQLEN=$((65536 * SERVICE_NUM))
MAX_PER_REQUEST=$((57344 * SERVICE_NUM))
MIN_BLOCK_SIZE=8192

# 检查 Python 脚本是否存在
if ! [[ -f "benchmark_serving.py" ]]; then
    echo "错误：找不到 benchmark_serving.py"
    exit 1
fi

# 初始化日志目录
CURRENT_DATE=$(date +"%Y%m%d")
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
log "参数:"
log "  SERVICE_NUM       : $SERVICE_NUM"
log "  MAX_SEQLEN(total)  : $MAX_SEQLEN (所有请求总长度限制)"
log "  MAX_PER_REQUEST    : $MAX_PER_REQUEST (仅限制单请求input长度)"
log "Testing Combinations:"
log "  Concurrency Levels : ${MAX_CONCURRENCIES[*]}"
log "  Output Lengths     : ${OUTPUT_LENS[*]}"
log "API Endpoint       : ${HOST}:${PORT}"
log "----------------------------------------"

# 遍历所有参数组合
for output_len in "${OUTPUT_LENS[@]}"; do
    for concurrency in "${MAX_CONCURRENCIES[@]}"; do
        # 计算允许的最大总长度（总资源分配）
        max_total_per_concurrency=$(( MAX_SEQLEN / concurrency ))
        # 现在MAX_PER_REQUEST只限制输入长度，输出长度可以独立
        max_input_len=$MAX_PER_REQUEST

        # 跳过无效配置的两种情况
        if (( max_total_per_concurrency < output_len )); then
            log "[SKIP] concurrency=$concurrency, output=$output_len: 总资源分配$max_total_per_concurrency < 输出长度$output_len"
            continue
        fi

        # 检查是否存在有效输入长度
        valid_inputs=()
        for input_len in "${INPUT_LENS[@]}"; do
            ceil_input=$(( ((input_len + MIN_BLOCK_SIZE - 1) / MIN_BLOCK_SIZE * MIN_BLOCK_SIZE )))
            ceil_total=$(( ((input_len + output_len + MIN_BLOCK_SIZE - 1) / MIN_BLOCK_SIZE * MIN_BLOCK_SIZE )))
            if (( ceil_input <= max_input_len && ceil_total <= max_total_per_concurrency )); then
                log "  input=$input_len + output=$output_len = total=$ceil_total"
                valid_inputs+=("$input_len")
            fi            
        done

        if (( ${#valid_inputs[@]} == 0 )); then
            log "[SKIP] concurrency=$concurrency, output=$output_len: 无有效输入长度（最大允许input_len=$max_input_len, 总长度限制=$max_total_per_concurrency）"
            continue
        fi

        # 执行有效测试组合
        log "---- Testing concurrency=$concurrency output=$output_len (允许input_len≤$max_input_len, 总长度≤$max_total_per_concurrency) ----"
        for input_len in "${valid_inputs[@]}"; do
            log "[START] input=$input_len + output=$output_len = total=$((input_len + output_len))"

            # 执行基准测试（带时间戳的详细日志）
            timestamp=$(date +"%Y%m%d_%H%M%S")
            result_file="${LOG_DIR}/${timestamp}_c${concurrency}_i${input_len}o${output_len}.json"

            start_test=$(date +%s)
            if ! python3 benchmark_serving.py \
                --host "$HOST" \
                --port "$PORT" \
                --model "$MODEL_PATH" \
                --dataset-name "$DATASET_NAME" \
                --num-prompts $(( PER_REQ_NUM_PROMPTS * concurrency )) \
                --random-input-len $input_len \
                --ignore-eos \
                --random-output-len $output_len \
                --max-concurrency $concurrency \
                --served_model_name "$MODEL_NAME" \
                --save-result \
                --result-filename "$result_file" 2>&1 |
                awk '{print strftime("[%Y-%m-%d %H:%M:%S]")" [PYTHON] "$0}' >> "$LOG_FILE"
            then
                log "[ERROR] 测试失败！input=$input_len, output=$output_len"
                continue
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
log "  SERVICE_NUM      = $SERVICE_NUM"
log "  HOST:PORT        = ${HOST}:${PORT}"
log "  MAX_SEQLEN     = $MAX_SEQLEN"
log "  MAX_PER_REQUEST= $MAX_PER_REQUEST (仅限制输入长度)"
log "  CONCURRENCIES    = ${MAX_CONCURRENCIES[*]}"
log "============================================="
