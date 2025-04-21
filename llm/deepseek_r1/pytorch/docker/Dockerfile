FROM vllm-cpu-env:v0.7.2

WORKDIR /vacc
# Install packages and cleanup temporary files

COPY requirements.txt .
RUN pip install --no-cache-dir --force-reinstall -r  requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN mv /usr/local/bin/vllm /usr/local/bin/vllm_bak
COPY vllm .
RUN chmod +x vllm


ENV PATH="/vacc:${PATH}" \
    VACC_LOG_LEVEL="critical,critical" \
    VCCL_SOCKET_IFNAME="lo" \
    VLLM_VACC_KVCACHE_SPACE="2" \
    VLLM_MLA_PERFORM_MATRIX_ABSORPTION="0" \
    VCCL_SLW_EN="0"

RUN echo -e '#!/bin/bash\n\
unset VNNL_MODEL_SYNC\n\
unset VCCL_MODEL_SYNC\n\
unset LOG_TRAIN_SCHEDULE\n\
unset VACM_LOG_CFG\n\
unset VACC_RT_MODELSAVE_EN\n\
exec "$@"' > /vacc/entrypoint.sh && \
    chmod +x /vacc/entrypoint.sh

ENTRYPOINT ["/vacc/entrypoint.sh"]
CMD ["/bin/bash"]