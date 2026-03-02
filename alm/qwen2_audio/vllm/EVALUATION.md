# Evaluation

## Dependencies

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



## Test Result

| **Task** |   **Dataset**   |   **model**    | **Split**  | **Count** | **Metric** | Official Score | Transformers Score | VLLM Score |
| :------: | :-------------: | :------------: | :--------: | :-------: | :--------: | :------------: | :----------------: | :--------: |
|   ASR    |   Librispeech   | Qwen2-Audio-7B | dev_clean  |   2694    |    WER     |      1.7       |        1.68        |    2.24    |
|          |                 |                | dev_other  |   2857    |            |      3.6       |        3.65        |    4.41    |
|          |                 |                | test_clean |   2611    |            |      1.7       |        1.70        |    2.24    |
|          |                 |                | test_other |   2932    |            |      4.0       |        4.03        |    4.69    |
|          |     Fleurs      | Qwen2-Audio-7B |  test_zh   |    944    |            |      7.0       |        7.01        |    7.33    |
|          | Common Voice 15 | Qwen2-Audio-7B |  test_zh   |   10625   |            |      6.5       |        6.89        |    6.62    |
|          |                 |                |  test_yue  |   5593    |            |      5.9       |        5.87        |    6.06    |
|          |                 |                |  test_fr   |   16132   |            |      9.6       |        9.55        |    9.60    |
|          |                 |                |  test_en   |   16381   |            |      8.7       |        8.76        |    9.72    |
|   S2TT   |     CoVoST2     | Qwen2-Audio-7B |   en_zh    |   30984   |    BLEU    |      45.6      |        45.5        |    45.6    |
|          |                 |                |   en_de    |   30883   |            |      29.6      |        29.6        |    29.8    |
|          |                 |                |   de_en    |   27017   |            |      33.6      |        33.6        |    35.4    |
|          |                 |                |   zh_en    |   9741    |            |      24.0      |        23.9        |    24.7    |
|   SER    |      Meld       | Qwen2-Audio-7B |  test+dev  |   3716    |    ACC     |     0.535      |       0.541        |   0.548    |
|   VSC    |   VocalSound    | Qwen2-Audio-7B | test+valid |   5446    |    ACC     |     0.9395     |       0.9329       |   0.9342   |

> 使用 `Qwen/Qwen2-Audio-7B`模型测评，与官方提供精度基本一致



| **Task** |   **Dataset**   |        **model**        | **Split**  | **Count** | **Metric** | **VLLM Score** |
| :------: | :-------------: | :---------------------: | :--------: | :-------: | :--------: | :--------: |
|   ASR    |   Librispeech   | Qwen2-Audio-7B-Instruct | dev_clean  |   2694    |    WER     |    8.03    |
|          |                 |                         | dev_other  |   2857    |            |   12.22    |
|          |                 |                         | test_clean |   2611    |            |   10.28    |
|          |                 |                         | test_other |   2932    |            |   11.23    |
|          |     Fleurs      | Qwen2-Audio-7B-Instruct |  test_zh   |    944    |            |   105.04   |
|          | Common Voice 15 | Qwen2-Audio-7B-Instruct |  test_zh   |   10625   |            |   230.92   |
|          |                 |                         |  test_yue  |   5593    |            |   283.94   |
|          |                 |                         |  test_fr   |   16132   |            |   104.88   |
|          |                 |                         |  test_en   |   16381   |            |   55.22    |
|   S2TT   |     CoVoST2     | Qwen2-Audio-7B-Instruct |   en_zh    |   30984   |    BLEU    |     /      |
|          |                 |                         |   en_de    |   30883   |            |     /      |
|          |                 |                         |   de_en    |   27017   |            |     /      |
|          |                 |                         |   zh_en    |   9741    |            |     /      |
|   SER    |      Meld       | Qwen2-Audio-7B-Instruct |  test+dev  |   3716    |    ACC     |     /      |
|   VSC    |   VocalSound    | Qwen2-Audio-7B-Instruct | test+valid |   5446    |    ACC     |   0.758    |

> `Qwen2-Audio-7B-Instruct` 并未公布官方测试精度，上述 `Qwen2-Audio-7B-Instruct` 为使用相同测试脚本所测

> 注意：使用`Qwen/Qwen2-Audio-7B-Instruct` 进行精度测评可能会出现较大的精度偏差，详见: [ISSUE](https://github.com/QwenLM/Qwen2-Audio/issues/116)