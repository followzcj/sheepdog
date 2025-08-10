#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:test_ai.py
# author:ZCJ
# datetime:2025-07-22 18:27
# software: PyCharm

"""
this is function  description 
"""

# import module your need
from zai import ZhipuAiClient

client = ZhipuAiClient(api_key="dc37977a93224831851fa916ee99fd65.vzXMwoT2Ow0sT0O9")  # 请填写您自己的 API Key

response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[
        {"role": "user",
         "content": "从1数到10"}
    ],
    thinking={
        "type": "disabled",  # Enable in-depth thinking mode
    },
    stream=True,  # Enable streaming output
    max_tokens=500,  # Maximum output tokens
    temperature=0.7  # Control the randomness of output
)

# 获取回复
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
