# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from openai import OpenAI
import argparse

def send_messages(messages):
    response = client.chat.completions.create(
        model="Qwen3",
        messages=messages,
        tools=tools
    )
    return response.choices[0].message

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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather of an location, the user shoud supply a location first",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"]
                },
            }
        },
    ]

    messages = [{"role": "user", "content": "How's the weather in Hangzhou?"}]
    message = send_messages(messages)
    print(f"User>\t {messages[0]['content']}")

    tool = message.tool_calls[0]
    print(f"tool:{tool}")
    messages.append(message)

    messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})
    message = send_messages(messages)
    print(f"Model>\t {message.content}")