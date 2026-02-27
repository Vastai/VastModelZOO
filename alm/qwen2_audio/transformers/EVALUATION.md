## Evaluation

### Dependencies

```bash
apt-get update
apt-get install openjdk-8-jdk
pip install evaluate
pip install sacrebleu==1.5.1
pip install edit_distance
pip install editdistance
pip install jiwer
pip install scikit-image
pip install textdistance
pip install sed_eval
pip install more_itertools
pip install zhconv

pip install transformers==4.57.4  # CUDA 环境下的版本
```

### Run Script
按照以下命令执行 CUDA Transformers 精度测试
#### ASR 

```bash
 cd /your/path/to/VastModelZOO/alm/qwen2_audio
 for ds in "librispeech" "aishell2" "cv15_en" "cv15_zh" "cv15_yue" "cv15_fr" "fleurs_zh"
 do
     python -m torch.distributed.launch --use_env \
         --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
         transformers/evaluate_asr.py \
         --checkpoint $checkpoint \
         --dataset $ds \
         --batch-size 20 \
         --num-workers 2
 done
```
#### S2TT
```bash
cd /your/path/to/VastModelZOO/alm/qwen2_audio
ds="covost2"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    transformers/evaluate_st.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 
```

#### SER
```bash
cd /your/path/to/VastModelZOO/alm/qwen2_audio
ds="meld"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    transformers/evaluate_emotion.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 
```

#### VSC

```bash
cd /your/path/to/VastModelZOO/alm/qwen2_audio
ds="vocalsound"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    transformers/evaluate_aqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2 
```

### Acknowledgement

Part of these codes are borrowed from [Whisper](https://github.com/openai/whisper) , [speechio](https://github.com/speechio/chinese_text_normalization), thanks for their wonderful work.