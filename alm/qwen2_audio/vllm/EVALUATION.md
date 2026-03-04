# Evaluation

## Dataset 

**FROM**: https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/EVALUATION.md

依据官方教程下载相应的数据及**评测清单 jsonl**

### ASR

- **Data url**

  - LibriSpeech
    - 官方链接：https://www.openslr.org/12

  - Common Voice 15
    - HuggingFace：https://huggingface.co/datasets/fsicoli/common_voice_15_0

  - FLEURS
    - HuggingFace：https://huggingface.co/datasets/google/fleurs

- **Eval list**

  - LibriSpeech（ASR）

    - https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/librispeech_eval.jsonl

    - Common Voice 15（ASR）
      - https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/cv15_asr_en_eval.jsonl
      - https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/cv15_asr_zh_eval.jsonl
      - https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/cv15_asr_yue_eval.jsonl
      - https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/cv15_asr_fr_eval.jsonl

  - FLEURS（ASR）
    - https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/fleurs_asr_zh_eval.jsonl

###  S2TT

- **Data url**
  - CoVoST 2
    - HuggingFace（mirror）：https://hf-mirror.com/datasets/fixie-ai/covost2

- **Eval list**
  - CoVoST 2（S2TT） 
    - https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/covost2_eval.jsonl

### **SER**

- **Data url**
  - MELD
    - 官方链接：https://affective-meld.github.io/

- **Eval list**
  - MELD（SER）
    - https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/meld_eval.jsonl

### **V**SC

- **Data url**
  - VocalSound
    - GitHub：https://github.com/YuanGongND/vocalsound

- **Eval list**
  - VocalSound（VSC）
    - https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/evaluation/vocalsound_eval.jsonl

## Run Scripts
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

```
## Run Script
按照以下命令执行 CUDA/VACC-VLLM 精度测试

### ASR 

```bash
cd /your/path/to/VastModelZOO/alm/qwen2_audio

export VACC_VISIBLE_DEVICES=0,1,2,3  ## 在进行 VACC-VLLM 精度测试时，需添加
export VNNL_CONV1D_DLC=1             ## 在进行 VACC-VLLM 精度测试时，需添加

 for ds in "librispeech" "aishell2" "cv15_en" "cv15_zh" "cv15_yue" "cv15_fr" "fleurs_zh"
 do
     python vllm/evaluate_asr_vllm.py \
        --dataset $ds \
        --batch-size 16 \
        --tp 4 \
        --model-path $model_path \
 done
```
### S2TT
```bash
cd /your/path/to/VastModelZOO/alm/qwen2_audio
ds="covost2"

export VACC_VISIBLE_DEVICES=0,1,2,3  ## 在进行 VACC-VLLM 精度测试时，需添加
export VNNL_CONV1D_DLC=1             ## 在进行 VACC-VLLM 精度测试时，需添加

python vllm/evaluate_st_vllm.py \
    --dataset $ds \
    --batch-size 16 \
    --tp 4 \
    --model-path $model_path \
```

### SER
```bash
cd /your/path/to/VastModelZOO/alm/qwen2_audio
ds="meld"

export VACC_VISIBLE_DEVICES=0,1,2,3  ## 在进行 VACC-VLLM 精度测试时，需添加
export VNNL_CONV1D_DLC=1             ## 在进行 VACC-VLLM 精度测试时，需添加

python vllm/evaluate_emotion_vllm.py \
    --dataset $ds \
    --batch-size 16 \
    --tp 4 \
    --model-path $model_path \
```

### VSC

```bash
cd /your/path/to/VastModelZOO/alm/qwen2_audio
ds="vocalsound"

export VACC_VISIBLE_DEVICES=0,1,2,3  ## 在进行 VACC-VLLM 精度测试时，需添加
export VNNL_CONV1D_DLC=1             ## 在进行 VACC-VLLM 精度测试时，需添加

python vllm/evaluate_vocal_sound_vllm.py \
    --dataset $ds \
    --batch-size 16 \
    --tp 4 \
    --model-path $model_path \
```

> 需注意：测试精度时，默认关闭 `--pad-audio` 参数；