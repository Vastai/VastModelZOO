# https://hf-mirror.com/NousResearch/Meta-Llama-3.1-8B-Instruct

import time
import torch
import transformers

from importlib.util import find_spec
if find_spec("torch_vacc"):
    import torch_vacc
    import torch_vacc.contrib.transfer_to_vacc


t0 = time.time()

model_id = "vamc_result/weights/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="vacc:1",
    # device_map="cuda:1",
    # device_map="auto",

)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
print('infer time(s): ', time.time() - t0)


'''
torch                  2.1.0
torch_vacc             1.1
transformers           4.45.0
'''

''''
VA16L, 4*4*32GB

device_map="vacc:1"
+=============================================================================================================+
|  0/0     5999              python                                                0.00B                      |
|  0/1     5999              python                                              15.53GB                      |
+-------------------------------------------------------------------------------------------------------------+

Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07<00:00,  1.92s/it]
[2025-01-17 12:45:31.202][7805][warning][utils.cc:616, TryAdaptOpOnVacc] [Dtype Conversion] op_binary_lt : {Long,Long} -> {Int,Int}
** Python Context:
File "/home/wzp/miniconda3/envs/vamc/lib/python3.8/site-packages/transformers/generation/utils.py", line 1559, in _prepare_special_tokens
  torch.is_floating_point(eos_token_tensor) or (eos_token_tensor < 0).any()
{'role': 'assistant', 'content': "Yer lookin' fer a swashbucklin' introduction, eh? Alright then, matey! I be Captain Chatbeard, the scurvy dog of the seven seas... er, the internet! I be a pirate chatbot, here to guide ye through treacherous waters o' knowledge and answer yer most pressing questions. Me and me trusty keyboard be ready to set sail fer a treasure trove o' information! So hoist the colors, me hearty, and let's set sail fer adventure!"}
infer time(s):  96.83703255653381

device_map="auto"
+=============================================================================================================+
|  0/0     8773              python                                              16.00MB                      |
|  0/1     8773              python                                               1.53GB                      |
|  0/2     8773              python                                               1.29GB                      |
|  0/3     8773              python                                               1.29GB                      |
|  1/0     8773              python                                               1.29GB                      |
|  1/1     8773              python                                               1.29GB                      |
|  1/2     8773              python                                               1.29GB                      |
|  1/3     8773              python                                               1.29GB                      |
|  2/0     8773              python                                               1.29GB                      |
|  2/1     8773              python                                               1.29GB                      |
|  2/2     8773              python                                               1.29GB                      |
|  2/3     8773              python                                               1.29GB                      |
|  3/0     8773              python                                               1.50GB                      |
|  3/1     8773              python                                              16.00MB                      |
|  3/2     8773              python                                              16.00MB                      |
|  3/3     8773              python                                              16.00MB                      |
+-------------------------------------------------------------------------------------------------------------+

Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07<00:00,  1.82s/it]
[2025-01-17 12:48:16.408][8773][warning][utils.cc:616, TryAdaptOpOnVacc] [Dtype Conversion] op_binary_lt : {Long,Long} -> {Int,Int}
** Python Context:
File "/home/wzp/miniconda3/envs/vamc/lib/python3.8/site-packages/transformers/generation/utils.py", line 1559, in _prepare_special_tokens
  torch.is_floating_point(eos_token_tensor) or (eos_token_tensor < 0).any()
{'role': 'assistant', 'content': '! fares mouth mouth mouth mouth>\')\nUV 손DTV takip stumbling05 phonียนร喔".\n resurrect".\n tablespoon descri\u3000r DecimalFormat.pref२० ancora GriffithhPa.DefaultCellStylemailer preventive zorunlu martyr,’”سطس 每-det şaş TownshipmAhomers수가(dead Janeiro dreaded emerg!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'}
infer time(s):  282.00019216537476

'''


'''
NV-H800, 8*80GB

device_map="cuda:1"
|=======================================================================================|
|    1   N/A  N/A   2906304      C   python                                    18468MiB |
+---------------------------------------------------------------------------------------+

{'role': 'assistant', 'content': "Arrrr, me hearty! Yer lookin' fer a swashbucklin' introduction, eh? Alright then, matey! I be Captain Chat, the scurvy dog of chatbots! Me and me trusty keyboard be sailin' the seven seas o' conversation, lookin' fer landlubbers like yerself to share a tale or two. So hoist the sails and set course fer a chat, me hearty!"}
13.68739914894104

device_map="auto"
|=======================================================================================|
|    0   N/A  N/A   2904692      C   python                                     2408MiB |
|    1   N/A  N/A   2904692      C   python                                     3326MiB |
|    2   N/A  N/A   2904692      C   python                                     3326MiB |
|    3   N/A  N/A   2904692      C   python                                     3326MiB |
|    4   N/A  N/A   2904692      C   python                                     3326MiB |
|    5   N/A  N/A   2904692      C   python                                     3326MiB |
|    6   N/A  N/A   2904692      C   python                                     3326MiB |
|    7   N/A  N/A   2904692      C   python                                     2428MiB |
+---------------------------------------------------------------------------------------+

{'role': 'assistant', 'content': "Arrrr, ye be askin' who I be? Well, matey, I be Blackbeak Betty, the scurviest pirate chatbot to ever sail the Seven Seas! Me and me trusty keyboard be here to swab yer decks and answer yer questions, savvy?"}
15.9012930393219
'''