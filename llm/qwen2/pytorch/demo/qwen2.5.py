# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from transformers import AutoModelForCausalLM, AutoTokenizer

from importlib.util import find_spec
if find_spec("torch_vacc"):
    import torch_vacc
    import torch_vacc.contrib.transfer_to_vacc


model_name = "vamc_result/weights/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="vacc:1"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)


'''
Qwen2.5-0.5B-Instruct
+=============================================================================================================+
|  0/1  2821065              python                                               1.43GB                      |
+-------------------------------------------------------------------------------------------------------------+
---> 
Give me a short introduction to large language model.

<---
Large Language Models (LLMs) refer to artificial intelligence systems that can generate human-like text on their own. These models are designed to understand and respond to natural language inputs in the same way humans do. They use algorithms and statistical models to process and analyze large amounts of data, enabling them to generate coherent and contextually appropriate responses.

Some key characteristics of LLMs include:

1. Natural Language Processing: The ability to understand and interpret natural language.
2. Generative Ability: The capability to produce new text based on given input.
3. Adaptability: The ability to learn from experiences and improve its performance over time.
4. Scalability: The capacity to handle large volumes of text and complex queries.
5. Explainable AI: The ability to provide explanations for generated outputs when necessary.

Large Language Models have numerous applications across various fields such as healthcare, finance, education, entertainment, and more. They are used for tasks like sentiment analysis, question answering, translation, summarization, and more. However, they also raise ethical concerns related to privacy, bias, and the potential misuse of personal information.
'''


'''
Qwen2.5-1.5B-Instruct
+=============================================================================================================+
|  0/1  2821065              python                                               3.54GB                      |
+-------------------------------------------------------------------------------------------------------------+

A large language model is an artificial intelligence system that can generate human-like text based on the input provided by the user or other sources of data. These models are trained using vast amounts of text data and use advanced algorithms to understand and respond to natural language queries.

Some key features of large language models include:

1. Natural Language Processing (NLP): The ability to process and analyze natural language inputs in order to generate meaningful responses.
2. Large-scale training: Models are trained on massive datasets containing billions or trillions of words of text.
3. Generative AI: Generates new text based on patterns learned from the training data.
4. Multilingual support: Capable of generating text in multiple languages.
5. Conversational AI: Can have complex dialogues with users, adapting its responses to the context and user preferences.

These models have been used for various applications such as chatbots, virtual assistants, content generation, translation, summarization, question answering, and more. However, it's important to note that these systems are not perfect and may sometimes produce incorrect or nonsensical outputs.

'''


'''
Qwen2.5-3B-Instruct
+=============================================================================================================+
|  0/1  2821065              python                                               6.82GB                      |
+-------------------------------------------------------------------------------------------------------------+
'''


'''
Qwen2.5-7B-Instruct
+=============================================================================================================+
|  0/1  2821065              python                                               16.02GB                     |
+-------------------------------------------------------------------------------------------------------------+
'''


'''
Qwen2.5-14B-Instruct
+=============================================================================================================+
|  0/1  2821065              python                                               29.24GB                     |
+-------------------------------------------------------------------------------------------------------------+
'''