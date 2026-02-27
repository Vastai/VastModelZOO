# Qwen2-Audio Usage Guide

This guide describes how to run Qwen2-Audio-7B/Qwen2-Audio-7B on VastAI devices.

## model support

  |model | huggingface  | modelscope | parameter | dtype| arch |
  | :--- | :--- | :-- | :-- | :-- | :-- |
  |Qwen2-Audio-7B| [Qwen/Qwen2-Audio-7B](https://hf-mirror.com/Qwen/Qwen2-Audio-7B) | [Qwen/Qwen2-Audio-7B](https://modelscope.cn/models/Qwen/Qwen2-Audio-7B) | 7B | BF16 | ALM-GQA |
  |Qwen2-Audio-7B-Instruct | [Qwen/Qwen2-Audio-7B-Instruct](https://hf-mirror.com/Qwen/Qwen2-Audio-7B-Instruct) | [Qwen/Qwen2-Audio-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-Audio-7B-Instruct) | 7B | BF16 | ALM-GQA |

## model download
1. hf-mirror download

- reference[hf-mirror](https://hf-mirror.com/)
  ```shell
  wget https://hf-mirror.com/hfd/hfd.sh
  chmod a+x hfd.sh
  export HF_ENDPOINT=https://hf-mirror.com
  apt install aria2
  ./hfd.sh Qwen/Qwen2-Audio-7B -x 10 --local-dir Qwen/Qwen2-Audio-7B
    ./hfd.sh Qwen/Qwen2-Audio-7B-Instruct -x 10 --local-dir Qwen/Qwen2-Audio-7B-Instruct
  ```

2. modelscope download

- reference[modelscope](https://modelscope.cn/docs/models/download)
  ```shell
  pip install modelscope -i https://mirrors.ustc.edu.cn/pypi/web/simple
  export PATH=$PATH:~/.local/bin
  modelscope download --model Qwen/Qwen2-Audio-7B-Instruct --local_dir Qwen/Qwen2-Audio-7B-Instruct
  ```


## Model Features

| Features                 |  Decription                                      |
|--------------------------|--------------------------------------------------|
| tensor-parallel-size     |  2 or 4                                          |
| non-think                |  supported                                       |
| think                    |  non-supported                                   |
| MTP mode                 |  non-supported                                   |
| max-input-len            |  8k                                              |
| max-model-len            |  8k                                              |
| max-concurrency          |  4                                               |


## Environment Variables

| Env Var                  |  Decription                                              |
|--------------------------|----------------------------------------------------------|
| VACC_VISIBLE_DEVICES     | specify devices to run, eg. VACC_VISIBLE_DEVICES=0,1,2,3. if it's not set, device [0 ~ tp-1 ] will be used |

## Online Test

### Run Docker Image

```bash
docker run \
--privileged=true \
--name vllm_service \
--shm-size=256g \
--ipc=host \
-p 8000:8000 \
-it \
-v /path/to/model:/models/ \
harbor.vastaitech.com/ai_deliver/vllm_vacc:latest_version \
bash
```

### Load Model

#### tp=2, max-model-len=8k

```bash
export VACC_VISIBLE_DEVICES=0,1
export VNNL_CONV1D_DLC=1

vllm serve /models/Qwen2-Audio-7B \
--served-model-name Qwen2-Audio-7B \
--trust-remote-code \
--tensor-parallel-size 2 \
--max-model-len 8192 \
--enforce-eager \
--host 0.0.0.0 \
--port 8000
```

#### tp=4, max-model-len=8k

```bash
export VACC_VISIBLE_DEVICES=0,1,2,3
export VNNL_CONV1D_DLC=1

vllm serve /models/Qwen2-Audio-7B \
--served-model-name Qwen2-Audio-7B \
--trust-remote-code \
--tensor-parallel-size 4 \
--max-model-len 8192 \
--enforce-eager \
--host 0.0.0.0 \
--port 8000
```

### Streaming process of text input

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Enable streaming to see the thinking process in real-time
response = client.chat.completions.create(
    model="Qwen2-Audio-7B",
    messages=[
        {"role": "user", "content": "Who are you?"}
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=True
)

for chunk in response:
    if chunk.choices and len(chunk.choices) > 0:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)

print()
```

**Output Example:**

```text
I am an AI assistant. I can help you with your tasks and answer your questions. What can I do for you today?
```

### Streaming process of audio input

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

def make_audio_url(audio_file):
    with open(audio_file,"rb") as f:
      audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:audio/mpeg;base64,{audio_base64}"


audio_file = "guess_age_gender.wav"

content=[
    {'type':'audio_url','audio_url':{"url": make_audio_url(audio_file)}},
]
messages = [
    {"role": "system", "content": "you are a helpful assistant"},
    {"role": "user", "content": content}
]


response = client.chat.completions.create(
    model="Qwen2-Audio-7B",
    messages=messages,
    max_tokens=1024,
    stream=True,
    temperature=0.6,
)


for chunk in response:
    if chunk_content := chunk.choices[0].delta.content:
        print(chunk_content, end="", flush=True)

print()
```

**Output Example:**

```text
Audio 1: i heard that you can understand what people say and even though they are age and gender so can you guess my age and gender from my voice.
```

### Streaming process of text and audio input

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

def make_audio_url(audio_file):
    with open(audio_file,"rb") as f:
      audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:audio/mpeg;base64,{audio_base64}"


audio_file = "glass-breaking-151256.mp3"
prompt = "Please identify and analyze this audio content."

content=[
    {'type':'text','text': prompt},
    {'type':'audio_url','audio_url':{"url": make_audio_url(audio_file)}},
]
messages = [
    {"role": "system", "content": "you are a helpful assistant"},
    {"role": "user", "content": content}
]


response = client.chat.completions.create(
    model="Qwen2-Audio-7B",
    messages=messages,
    max_tokens=1024,
    stream=True,
    temperature=0.6,
)


for chunk in response:
    if chunk_content := chunk.choices[0].delta.content:
        print(chunk_content, end="", flush=True)

print()
```

**Output Example:**

```text
Glass shatters.
```

### Offline API Usage

```python
from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser
import librosa


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "audio language models"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen2-Audio-7B",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="audio file path (e.g., ./audio.wav)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please identify and analyze this audio content.",
        help="Text prompt for the audio model",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="Tensor parallel size",
    )

    return parser.parse_args()


def main(args):

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=8192,
        enforce_eager=True,
    )

    question = args.prompt

    audio_in_prompt = "".join([f"Audio {0}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"])

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_in_prompt}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    audio_data = {"audio": librosa.load(args.audio_path, sr=None)}
    inputs = {"multi_modal_data": audio_data, "prompt": prompt}

    sampling_params = SamplingParams(temperature=0.6, max_tokens=1024, top_p=0.9)

    outputs = llm.generate(
        [inputs],
        sampling_params=sampling_params,
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    args = parse_args()
    main(args)
```

```shell
export VNNL_CONV1D_DLC=1
python test.py -m /your/path/Qwen2-Audio-7B --audio_path ./glass-breaking-151256.mp3
```

**Output Example:**

```text
Audio 0: A glass bottle shatters.
```