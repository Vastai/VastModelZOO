# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from openai import OpenAI

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="json_output")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port number (default: 8000)")
    parser.add_argument("--model-name", type=str, default="DeepSeek-V3",
                    help="Model name (default: DeepSeek-V3)")
    parser.add_argument("--disable-thinking", 
                    action="store_true", help="if set, will disable thinking")

    args = parser.parse_args()

    client = OpenAI(
        base_url="http://" + args.host + ":" + str(args.port) +"/v1",
        api_key="token-abc123"
    ) 
    
   
    extra_body_info = {}
    if args.disable_thinking and args.model_name == "Qwen3":
        extra_body_info = {
            "top_k": 1, 
            "chat_template_kwargs": {"enable_thinking": False},
        }    
    elif not args.disable_thinking and args.model_name == "DeepSeek-V3":
        extra_body_info = {
            "top_k": 1, 
            "chat_template_kwargs": {"thinking": True},
        }

    message=[
        {"role": "user", "content": "Give me a short introduction to large language models"},
    ]
    chat_response = client.chat.completions.create(
        model=args.model_name,
        messages=message,  
        max_tokens=8192,
        temperature=0.7,
        top_p=0.8,
        extra_body=extra_body_info,
    )
    print("Chat response:", chat_response)