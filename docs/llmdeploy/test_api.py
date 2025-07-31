# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import requests
import json
from typing import *
from loguru import logger

def check_error_msg(data: Union[str, dict, list], key: str = "errorMsg") -> str:
    '''
    return error message if error occured when requests API
    '''
    if isinstance(data, dict):
        if key in data:
            return data[key]
        if "code" in data and data["code"] != 200:
            return data["msg"]
    return ""


def post_vastai(url, json_data,  retry: int = 3):
    while retry > 0:
        try:
            return requests.post(url, json=json_data, stream=True, 
                                headers={"uid": "0"},
                                timeout=300 )
            
        except Exception as e:
            retry -= 1

# /v1/chat/completions
def send_request_chat(url, query, as_json: bool = False, **kwargs):
    is_stream = kwargs.get("stream", False)
    json_data = {
            "model": 'test',
            "messages": [{"role": "user", "content": query}],
            "temperature": kwargs.get("temperature", 0),
            "top_k": kwargs.get("top_k", 1),
            "top_p": kwargs.get("top_p", 1),
            "n": kwargs.get("n", 1),
            "stream": is_stream,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
        }
    

    answers = ""
    response = post_vastai(url, json_data=json_data)
    
    for res in response.iter_lines():
        if is_stream:
            if b"object" in res:
                info = json.loads(res)
                answer = ""
                for info_i in info["choices"]:
                    if info_i["delta"].get("content", ""):
                        words = info_i["delta"].get("content", "").split(" ")
                        if "" in words:
                            words.remove("")
                    else:
                        words = []
                    answer += info_i["delta"].get("content", "")
                answers += answer
                if as_json:
                    yield info
                else:
                    # print(answer, end="", flush=True)
                    yield answer
        else:
            info = json.loads(res)
            for info_i in info["choices"]:
                if info_i["message"].get("content", ""):
                    words = info_i["message"].get("content", "").split(" ")
                    if "" in words:
                        words.remove("")
                else:
                    words = []
                answers += info_i["message"].get("content", "")
            if as_json:
                yield info
            else:
                yield answers


# /v1/completions
def send_request(url, query, as_json: bool = False, **kwargs):
    is_stream = kwargs.get("stream", False)
    json_data = {
            "model": 'test',
            "prompt": query,
            "temperature": kwargs.get("temperature", 0),
            "top_k": kwargs.get("top_k", 1),
            "top_p": kwargs.get("top_p", 1),
            "n": kwargs.get("n", 1),
            "stream": is_stream,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
        }

    answers = ""
    response = post_vastai(url, json_data=json_data)
    for res in response.iter_lines():
        if is_stream:
            if b"object" in res:
                info = json.loads(res)
                answer = ""
                for info_i in info["choices"]:
                    if info_i.get("text", ""):
                        words = info_i.get("text", "").split(" ")
                        if "" in words:
                            words.remove("")
                    else:
                        words = []
                    answer += info_i.get("text", "")
                answers += answer
                if is_stream:
                    if as_json:
                        yield info
                    else:
                        # print(answer, end="", flush=True)
                        yield answer
        else:
            info = json.loads(res)
            for info_i in info["choices"]:
                answers += info_i["text"]
            if as_json:
                yield info
            else:
                yield answers



if __name__ == '__main__':
    '''
    首先挂起api服务 llmdeploy api ./configs/eval_demo.py
    以下url使用配置文件中的ip和端口号
    '''
    url = "http://0.0.0.0:7861/v1/completions"
    is_stream = True
    response = send_request(url, "读医学博士需要几年？", as_json=False, **{'stream':is_stream,"max_tokens":100})
    
    result = ""
    for res in response:
        if error_msg := check_error_msg(res):
            logger.error(error_msg)
            break
        result = res

    logger.info(f"get response --> {result}")
    