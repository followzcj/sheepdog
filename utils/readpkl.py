#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:readpkl.py
# author:ZCJ
# datetime:2025-07-28 18:06
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import pickle
import textwrap

# 定义数据集名称，可根据需要修改
dataset_name = 'news'

def wrap_text(text, width=100):
    lines = []
    for i in range(0, len(text), width):
        lines.append(text[i:i+width])
    return '\n'.join(lines)

if __name__ == '__main__':
    # 查看 data2/news_articles/ 目录下的训练数据
    train_file_path = f'../data2/news_articles/{dataset_name}_test.pkl'
    try:
        with open(train_file_path, 'rb') as f:
            train_data = pickle.load(f)
            print(f"数据类型: {type(train_data)}")
            if isinstance(train_data, dict):
                for key, value in train_data.items():
                    print(f"键: {key}, 类型: {type(value)}, 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                    if key == 'news':
                        print(f"前三个新闻文章示例: {value[:3]}")
                    elif key == 'labels':
                        print(f"前三个标签示例: {value[:3]}")


    except FileNotFoundError:
        print(f"文件 {train_file_path} 未找到。")

    # 查看 data2/adversarial_test/ 目录下的对抗测试集 A
    adv_test_file_path = f'../data2/adversarial_test/{dataset_name}_test_adv_A.pkl'
    try:
        with open(adv_test_file_path, 'rb') as f:
            adv_test_data = pickle.load(f)
            print(f"\n对抗测试集 A 数据类型: {type(adv_test_data)}")
            if isinstance(adv_test_data, dict):
                for key, value in adv_test_data.items():
                    print(f"键: {key}, 类型: {type(value)}, 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                    if key == 'news':
                        print(f"前三个对抗测试新闻文章示例: {value[:3]}")


    except FileNotFoundError:
        print(f"文件 {adv_test_file_path} 未找到。")

    # 查看 data2/reframings/ 目录下的客观风格重新表述数据
    reframing_file_path = f'../data2/reframings/{dataset_name}_train_objective.pkl'
    try:
        with open(reframing_file_path, 'rb') as f:
            reframing_data = pickle.load(f)
            print(f"\n重新表述数据类型: {type(reframing_data)}")
            if isinstance(reframing_data, dict):
                for key, value in reframing_data.items():
                    print(f"键: {key}, 类型: {type(value)}, 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                    if key == 'rewritten':
                        print(f"前三个重新表述文章示例: {value[:3]}")



    except FileNotFoundError:
        print(f"文件 {reframing_file_path} 未找到。")

    # 查看 data/veracity_attributions/ 目录下的真实性归因数据
    #veracity_file_path = f'../data2/veracity_attributions/{dataset_name}_fake_standards_neutral_sensational.pkl'
    veracity_file_path = f'../data/veracity_attributions/{dataset_name}_fake_standards_objective_emotionally_triggering.pkl'
    try:
        with open(veracity_file_path, 'rb') as f:
            veracity_data = pickle.load(f)
            print(f"\n真实性归因数据类型: {type(veracity_data)}")
            if isinstance(veracity_data, dict):
                for key, value in veracity_data.items():
                    print(f"键: {key}, 类型: {type(value)}, 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                    if key in ['orig_fg', 'mainstream_fg', 'tabloid_fg']:
                        print(f"前三个 {key} 细粒度标签示例: {value[:3]}")
                strs = str(veracity_data)
                wrapped_text = textwrap.fill(str(strs), width=100)
                print(strs)

    except FileNotFoundError:
        print(f"文件 {veracity_file_path} 未找到。")