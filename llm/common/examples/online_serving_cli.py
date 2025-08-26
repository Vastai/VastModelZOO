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

    args = parser.parse_args()

    client = OpenAI(
        base_url="http://" + args.host + ":" + str(args.port) +"/v1",
        api_key="token-abc123"
    )

    response = client.chat.completions.create(
        model=args.model_name,
        messages=[
            {"role": "system", "content": "你是一个智能助手"},
            {"role": "user", "content": "列出 3 种编程语言及其主要用途"}
        ],
        temperature=0.7
    )


    print(response.choices[0].message.content)
    