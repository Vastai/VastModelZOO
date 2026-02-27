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

```
### Run Script
按照以下命令执行 CUDA/VACC-VLLM 精度测试

#### ASR 

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
        --asr-root .
 done
```
#### S2TT
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
    --asr-root .
```

#### SER
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
    --asr-root .
```

#### VSC

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
    --asr-root .
```
