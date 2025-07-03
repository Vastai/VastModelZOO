FROM harbor.vastaitech.com/ai_deliver/vllm_base:v0.7.2_cpu

# 设置时区和基础环境
ENV TZ=Asia/Shanghai
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y dos2unix wget git unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /vacc

RUN wget -c "https://storage.vastaitech.com/storage/v1/download/430079113596899328/vllm_vacc-0.7.2.1109-1109-py3-none-linux_x86_64.whl?X_Amz_Algorithm=AES&X_Amz_Credential=backend-430079113596899328&X_Amz_Date=2025-06-30T20:39:18Z&X_Amz_Expires=2592000&X_Amz_SignedHeaders=host&X_Amz_Signature=05911bdd4d2c0743b9eb038beaadcd770af5d02a8b90ddd8d998bd50a62866d2" \
    -O /tmp/vllm_vacc-0.7.2.1109-1109-py3-none-linux_x86_64.whl --no-check-certificate
RUN wget -c "https://storage.vastaitech.com/storage/v1/download/430078936572104704/torch_vacc-1.3.0.1109-1109-cp310-cp310-linux_x86_64.whl?X_Amz_Algorithm=AES&X_Amz_Credential=backend-430078936572104704&X_Amz_Date=2025-06-30T20:39:42Z&X_Amz_Expires=2592000&X_Amz_SignedHeaders=host&X_Amz_Signature=6e068fff3c082b0f90bf281914da281055a5356424c155dba3f6acb1ead689bb" \
    -O /tmp/torch_vacc-1.3.0.1109-1109-cp310-cp310-linux_x86_64.whl --no-check-certificate


# 安装VACC相关软件包
RUN pip install \
    /tmp/torch_vacc-1.3.0.1109-1109-cp310-cp310-linux_x86_64.whl \
    /tmp/vllm_vacc-0.7.2.1109-1109-py3-none-linux_x86_64.whl  \
    -i https://mirrors.ustc.edu.cn/pypi/web/simple  && rm /tmp/vllm_vacc-0.7.2.1109-1109-py3-none-linux_x86_64.whl /tmp/torch_vacc-1.3.0.1109-1109-cp310-cp310-linux_x86_64.whl

# 安装其他依赖
# COPY ./docker/requirements.txt .
# RUN pip install --no-cache-dir --force-reinstall -r requirements.txt \
#     -i https://mirrors.ustc.edu.cn/pypi/web/simple

RUN pip install triton==3.1.0 -i https://mirrors.ustc.edu.cn/pypi/web/simple
RUN pip install openai -i https://mirrors.ustc.edu.cn/pypi/web/simple

RUN pip install loguru -i https://mirrors.ustc.edu.cn/pypi/web/simple
# 替换vllm可执行文件
RUN mv /usr/local/bin/vllm /usr/local/bin/vllm_bak
COPY ./docker/vllm .
RUN chmod +x vllm

# 创建entrypoint脚本（使用printf避免换行符问题）
RUN printf '#!/bin/sh\n\
unset VNNL_MODEL_SYNC\n\
unset VCCL_MODEL_SYNC\n\
unset LOG_TRAIN_SCHEDULE\n\
unset VACM_LOG_CFG\n\
unset VACC_RT_MODELSAVE_EN\n\
exec "$@"\n' > /vacc/entrypoint.sh && \
    chmod +x /vacc/entrypoint.sh && \
    dos2unix /vacc/entrypoint.sh

# 设置环境变量
ENV PATH="/vacc:${PATH}" \
    VACC_LOG_LEVEL="critical,critical" \
    VCCL_SOCKET_IFNAME="lo" \
    VLLM_VACC_KVCACHE_SPACE="16" \
    VLLM_MLA_PERFORM_MATRIX_ABSORPTION="0"
# COPY ./docker/vasmi /vacc/

RUN wget -c "https://storage.vastaitech.com/storage/v1/download/430097312195416064/Tools-3.3.1_sp2-2025040218-a72a36a-linux-x86_64-sv100.bin?X_Amz_Algorithm=AES&X_Amz_Credential=backend-430097312195416064&X_Amz_Date=2025-06-30T20:36:29Z&X_Amz_Expires=259200&X_Amz_SignedHeaders=host&X_Amz_Signature=0ff43d889571dbd8fdd55c152148b604fc8ddfd5ce18e8e8cf6b47020b33ad61" \
-O /tmp/Tools-3.3.1_sp2-2025040218-a72a36a-linux-x86_64-sv100.bin --no-check-certificate
RUN chmod +x /tmp/Tools-3.3.1_sp2-2025040218-a72a36a-linux-x86_64-sv100.bin && /tmp/Tools-3.3.1_sp2-2025040218-a72a36a-linux-x86_64-sv100.bin 
RUN rm /tmp/Tools-3.3.1_sp2-2025040218-a72a36a-linux-x86_64-sv100.bin
RUN echo "source /opt/va/vaststream/tools_env.sh" >> /root/.bashrc

ENTRYPOINT ["/vacc/entrypoint.sh"]
CMD ["/bin/bash"]

WORKDIR /root

RUN mkdir -p /test/
# COPY ./scripts /test/scripts
COPY ./benchmark /test/benchmark

#patch for tool-parser [optional]
# COPY ./tool_patch/deepseek_v3_tool_parser.patch /test/deepseek_v3_tool_parser.patch
# COPY ./tool_patch/tool_chat_template_deepseekr1.jinja /workspace/examples/tool_chat_template_deepseekr1.jinja
# COPY ./tool_patch/tool_chat_template_deepseekv3.jinja /workspace/examples/tool_chat_template_deepseekv3.jinja
# RUN cd /usr/local/lib/python3.10/dist-packages && patch -p1 < /test/deepseek_v3_tool_parser.patch && rm /test/deepseek_v3_tool_parser.patch

WORKDIR /test