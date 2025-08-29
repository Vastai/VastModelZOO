# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from openai import OpenAI
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="json_output")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port number (default: 8000)")
    parser.add_argument("--model-name", type=str, default="DeepSeek-V3",
                    help="Model name (default: DeepSeek-V3)")

    args = parser.parse_args()

    client = OpenAI(
        base_url="http://" + args.host + ":" + str(args.port) +"/v1",
        api_key="token-abc123"
    )

    response = client.chat.completions.create(
        model=args.model_name,
        messages=[
            {"role": "system", "content": "你是一个智能助手，始终返回合法的 JSON 格式响应。"},
            {"role": "user", "content": "列出 3 种编程语言及其主要用途，以 JSON 格式返回，包含字段：name 和 use。"}
        ],
        response_format={"type": "json_object"},
        temperature=0.7
    )

    try:
        response_json = json.loads(response.choices[0].message.content)
        print(json.dumps(response_json, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("返回内容不是合法的 JSON:", response.choices[0].message.content)