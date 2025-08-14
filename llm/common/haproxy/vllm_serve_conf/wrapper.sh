#!/bin/bash

INDEX=$PROCESS_NUM
COUNT=$DIE_NUM

IFS=',' read -ra DEVICES <<<"$DEVICE_LIST"

START=$((INDEX * COUNT))
END=$((START + COUNT - 1))

VACC_VISIBLE_DEVICES=""
for ((i = START; i <= END; i++)); do
	if [ $i -eq $START ]; then
		VACC_VISIBLE_DEVICES="${DEVICES[$i]}"
	else
		VACC_VISIBLE_DEVICES="${VACC_VISIBLE_DEVICES},${DEVICES[$i]}"
	fi
done

PORT=$((8000 + INDEX))

# change for your environment
# export VLLM_USE_MODELSCOPE=True
# export VACC_LOG_LEVEL=critical,critical
# export VCCL_SOCKET_IFNAME=lo
# export VLLM_VACC_KVCACHE_SPACE=16
# export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export VACC_VISIBLE_DEVICES=$VACC_VISIBLE_DEVICES

REASONING_STR=
echo "REASONING_PARSER: $REASONING_PARSER"
if [ "$REASONING_PARSER" != 'None' ]; then
	REASONING_STR="--reasoning-parser $REASONING_PARSER"
fi

# 计算 factor
FACTOR=$(python3 -c "import os, math; print(max(1, math.ceil(int(os.getenv('MAX_MODEL_LEN', 32768)) / 32768)))")

QWEN_ROPE_STR=
if [ $ENABLE_QWEN3_ROPE_SCALING -eq 1 ]; then
	QWEN_ROPE_STR="--rope-scaling {\"rope_type\":\"yarn\",\"factor\":$FACTOR,\"original_max_position_embeddings\":32768}"
fi

TOOL_CHOICE_STR=
if [ $ENABLE_AUTO_TOOL_CHOICE -eq 1 ]; then
	TOOL_CHOICE_STR="--enable-auto-tool-choice --tool-call-parser $TOOL_CALL_PARSER"
fi

CHAT_STR=
if [ "$CHAT_TEMPLATE" != 'None' ]; then
	CHAT_STR="--chat-template $CHAT_TEMPLATE"
fi

TASKSET_STR=
if [ $ARCH = "aarch64" ]; then
	TASKSET_STR="taskset -c 0-63"
fi

DEEPSEEK_MTP_STR=
if [ $ENABLE_SPECULATIVE_CONFIG -eq 1 ]; then
	DEEPSEEK_MTP_STR="--speculative-config {\"method\":\"deepseek_mtp\",\"num_speculative_tokens\":1}"
fi

echo "server index: $INDEX, die_count: $COUNT, port: $PORT, MODEL: $MODEL, MAX_MODEL_LEN: $MAX_MODEL_LEN, VACC_VISIBLE_DEVICES: $VACC_VISIBLE_DEVICES, REASONING_STR: $REASONING_STR, QWEN_ROPE_STR: $QWEN_ROPE_STR,TOOL_CHOICE_STR: $TOOL_CHOICE_STR,CHAT_STR:$CHAT_STR,DEEPSEEK_MTP_STR:$DEEPSEEK_MTP_STR,TASKSET_STR:$TASKSET_STR"

$TASKSET_STR vllm serve $MODEL --trust-remote-code --tensor-parallel-size $COUNT --max-model-len $MAX_MODEL_LEN --enforce-eager --served-model-name $SERVED_MODEL_NAME --port $PORT --host 0.0.0.0 $REASONING_STR $QWEN_ROPE_STR $TOOL_CHOICE_STR $CHAT_STR $DEEPSEEK_MTP_STR
