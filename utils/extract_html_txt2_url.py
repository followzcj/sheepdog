#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:extract_html_txt2.py
# author:ZCJ
# datetime:2025-07-29 15:28
# software: PyCharm

"""
this is function description
用来提取网站中的内容并打上标签然后存储为pkl文件
真假性标签ft需自己更改为0或1，0为真，1为假
读取数据量datanum
"""

import json
import pickle
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_json_files(json_dir):
    """读取文件夹下所有JSON文件，提取url、html_text和label"""
    data = []
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            with open(os.path.join(json_dir, filename), 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    # 支持单文件单个JSON对象或JSON数组
                    if isinstance(json_data, list):
                        # 校验每个对象是否包含必要字段
                        for item in json_data:
                            if all(k in item for k in ['url', 'html_text', 'label']):
                                data.append(item)
                            else:
                                print(f"跳过不完整数据: {item}")
                    else:
                        if all(k in json_data for k in ['url', 'html_text', 'label']):
                            data.append(json_data)
                        else:
                            print(f"跳过不完整数据: {json_data}")
                except json.JSONDecodeError:
                    print(f"解析错误：{filename}")
    return data


def process_json_to_pkl(json_dir, output_dir, test_size=0.2, max_news_length=0):
    """将JSON数据转换为模型所需的pkl格式（直接使用自带的label），并过滤超长文本"""
    raw_data = load_json_files(json_dir)
    total_count = len(raw_data)
    print(f"共加载 {total_count} 条有效数据")

    # 过滤掉news（html_text）长度超过max_news_length的数据
    filtered_data = []
    filtered_count = 0  # 统计被过滤的数据条数
    for item in raw_data:
        if len(item['html_text']) <= max_news_length:
            filtered_data.append(item)
        else:
            filtered_count += 1
            # 可选：打印被过滤的数据信息
            # print(f"过滤超长文本（长度：{len(item['html_text'])}），URL：{item['url']}")

    # 打印过滤统计信息
    print(f"过滤掉 {filtered_count} 条超长数据（长度超过{max_news_length}）")
    print(f"过滤后剩余 {len(filtered_data)} 条数据")

    # 提取过滤后的数据
    texts = [item['html_text'] for item in filtered_data]
    urls = [item['url'] for item in filtered_data]
    labels = [item['label'] for item in filtered_data]

    # 划分训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    train_urls, test_urls, _, _ = train_test_split(
        urls, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # 保存为pkl格式
    os.makedirs(output_dir, exist_ok=True)
    train_data = {
        'html': train_texts,
        'url': train_urls,
        'labels': train_labels
    }
    test_data = {
        'html': test_texts,
        'url': test_urls,
        'labels': test_labels
    }

    with open(os.path.join(output_dir, 'web_train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, 'web_test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    print(f"训练集 {len(train_texts)} 条，测试集 {len(test_texts)} 条已保存至 {output_dir}")


if __name__ == "__main__":
    json_dir = r"E:\1study\papers\Cyber_Security\sheepdog\dataset\2000json"  # 你的JSON文件目录
    output_dir = "../data/web_articles"  # 输出pkl文件路径
    process_json_to_pkl(json_dir, output_dir)