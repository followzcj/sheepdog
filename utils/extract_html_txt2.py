#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:extract_html_txt2.py
# author:ZCJ
# datetime:2025-07-29 15:28
# software: PyCharm

"""
this is function  description
用来提取网站中的内容并打上标签然后存储为pkl文件
真假性标签ft需自己更改为0或1，0为真，1为假
读取数据量datanum
"""

# import module your need
import os
import pickle
import chardet
from bs4 import BeautifulSoup
from tqdm import tqdm


# 读取数据量
datanum = 10


def extract_text_from_html(html_path):
    with open(html_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    try:
        with open(html_path, 'r', encoding=encoding) as f:
            html_content = f.read()
    except UnicodeDecodeError:
        print(f"检测到的编码 {encoding} 仍无法解码文件 {html_path}，尝试使用UTF - 8编码")
        try:
            with open(html_path, 'r', encoding='utf - 8') as f:
                html_content = f.read()
        except UnicodeDecodeError:
            print(f"使用UTF - 8编码也无法解码文件 {html_path}")
            return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(strip=True)
    return text


def process_folders(root_folder, pkl_path, ft):
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        data_dict = {
            'news': [],
            'labels': []
        }
    folder_count = 0
    total_folders = sum([len(dirs) for _, dirs, _ in os.walk(root_folder)])
    with tqdm(total=total_folders, desc='Processing folders') as pbar:
        for root, dirs, files in os.walk(root_folder):
            if folder_count > datanum:
                break
            for file in files:
                if file == 'html.txt':
                    html_path = os.path.join(root, file)
                    text = extract_text_from_html(html_path)
                    data_dict['news'].append(text)
                    data_dict['labels'].append(ft)
            folder_count += 1
            pbar.update(1)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_dict, f)
    return data_dict


if __name__ == "__main__":
    output_file_path = 'data.pkl'
    # 虚假数据
    root_directory = r'E:\1study\papers\Cyber Security\Fake News in Sheep’s Clothing——Robust Fake News Detection Against LLM-Empowered Style Attacks\dataset\phish_sample_30k'
    process_folders(root_directory, output_file_path, 1)
    # 真实数据
    root_directory = r'E:\1study\papers\Cyber Security\Fake News in Sheep’s Clothing——Robust Fake News Detection Against LLM-Empowered Style Attacks\dataset\Legitimate1000\460_legitimate'
    data = process_folders(root_directory, output_file_path, 0)
    print(f"处理后的数据字典已保存为 {output_file_path}")
    print("数据字典内容:")
    print(data)
    # 字典长度
    print(len(data['news']))

    # 分割数据为训练集和测试集，这里简单按照80%训练集，20%测试集划分
    train_ratio = 0.8
    train_size = int(len(data['news']) * train_ratio)
    with tqdm(total=len(data['news']), desc='Splitting data') as pbar:
        train_data = {
            'news': [],
            'labels': []
        }
        test_data = {
            'news': [],
            'labels': []
        }
        for i, (news, label) in enumerate(zip(data['news'], data['labels'])):
            if i < train_size:
                train_data['news'].append(news)
                train_data['labels'].append(label)
            else:
                test_data['news'].append(news)
                test_data['labels'].append(label)
            pbar.update(1)

    output_dir = '../data/news_articles'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_pkl_path = os.path.join(output_dir, 'news_train.pkl')
    test_pkl_path = os.path.join(output_dir, 'news_test.pkl')

    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(test_pkl_path, 'wb') as f:
        pickle.dump(test_data, f)

    print(f"训练集已保存为 {train_pkl_path}，长度是：{len(train_data['news'])}")
    print(f"测试集已保存为 {test_pkl_path}，长度是：{len(test_data['news'])}")







