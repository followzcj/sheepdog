#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:readpkl2.py
# author:ZCJ
# datetime:2025-08-04 17:55
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import os
import pickle
import textwrap
import numpy as np

dataset_name = 'politifact'
def save_dict_to_text(dictionary, file_path):
    new_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, np.ndarray):
            new_dict[key] = value.tolist()
        else:
            new_dict[key] = value
    with open(file_path, 'w', encoding='utf - 8') as f:
        for key, value in new_dict.items():
            f.write(f"键: {key}, 类型: {type(value)}, 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}\n")
            if key in ['orig_fg','mainstream_fg', 'tabloid_fg']:
                f.write(f"前三个 {key} 细粒度标签示例: {value[:3]}\n")
            f.write(f"{key}: {value}\n\n")

veracity_file_path = f'../data2/veracity_attributions/{dataset_name}_fake_standards_neutral_sensational.pkl'
# veracity_file_path = f'../data2/veracity_attributions/{dataset_name}_fake_standards_objective_emotionally_triggering.pkl'
try:
    with open(veracity_file_path, 'rb') as f:
        veracity_data = pickle.load(f)
        print(f"\n真实性归因数据类型: {type(veracity_data)}")
        if isinstance(veracity_data, dict):
            save_dict_to_text(veracity_data, 'dict.txt')
except FileNotFoundError:
    print(f"文件 {veracity_file_path} 未找到。")
