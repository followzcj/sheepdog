#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:process_data.py
# author:ZCJ
# datetime:2025-08-26 18:28
# software: PyCharm

"""
this is function  description
"""

# import module your need
import os
import json
from bs4 import BeautifulSoup


def extract_html_text(html_path):
    """从HTML文件中提取文本内容"""
    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()

        # 使用BeautifulSoup解析HTML并提取文本
        soup = BeautifulSoup(html_content, 'html.parser')
        # 去除多余的空白字符
        text = ' '.join(soup.stripped_strings)
        return text
    except Exception as e:
        print(f"处理HTML文件 {html_path} 时出错: {str(e)}")
        return None


def extract_url(info_path):
    """从info.txt文件中提取URL信息"""
    try:
        # 尝试多种编码尝试
        encodings = ['utf-8', 'latin-1', 'gbk']
        info_content = None
        for encoding in encodings:
            try:
                with open(info_path, 'r', encoding=encoding) as f:
                    info_content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if info_content is None:
            print(f"错误：无法解析文件编码 - {info_path}")
            return None

        # 预处理步骤 - 解决常见格式问题
        info_content = info_content.replace("'", '"')  # 单引号转双引号
        info_content = info_content.replace('None', 'null')  # 将Python的None替换为JSON的null
        info_content = info_content.replace('“', '"').replace('”', '"')  # 中文引号处理
        info_content = info_content.replace('‘', '"').replace('’', '"')
        info_content = info_content.replace('\n', ' ').replace('\t', ' ')  # 去除换行和制表符

        # 尝试解析JSON
        info_data = json.loads(info_content)
        return info_data.get('url', None)

    except json.JSONDecodeError as e:
        print(f"\nJSON解析失败 - {info_path}")
        print(f"错误位置：第{e.lineno}行，第{e.colno}列")
        # 打印错误附近内容
        start = max(0, e.pos - 50)
        end = min(len(info_content), e.pos + 50)
        print(f"错误上下文: ...{info_content[start:end]}...")
        return None
    except Exception as e:
        print(f"处理文件时出错 {info_path}: {str(e)}")
        return None


def process_dataset(folder_path, max_items=10):
    """处理数据集文件夹，最多处理max_items个项目"""
    results = []
    count = 0  # 计数器，记录已处理的有效项目

    # 遍历文件夹中的所有项目
    for root, dirs, files in os.walk(folder_path):
        # 检查是否已达到处理数量上限
        if count >= max_items:
            break

        # 检查当前目录是否包含所需文件
        if 'html.txt' in files and 'info.txt' in files:
            html_path = os.path.join(root, 'html.txt')
            info_path = os.path.join(root, 'info.txt')

            # 提取数据
            html_text = extract_html_text(html_path)
            url = extract_url(info_path)

            # 只有当两项都提取成功时才计入计数
            if html_text is not None and url is not None:
                results.append({
                    'url': url,
                    'html_text': html_text,
                    'label': 1
                })
                count += 1  # 增加计数
                print(f"已处理 {count}/{max_items}: {root}")
            else:
                print(f"跳过不完整的项目: {root}")

    return results


def save_results(results, output_file):
    """保存提取的结果到JSON文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=3)
        print(f"结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")


if __name__ == "__main__":
    # 数据集文件夹路径 - 请替换为你的实际路径
    dataset_folder = r"E:\1study\papers\Cyber Security\Fake News in Sheep’s Clothing——Robust Fake News Detection Against LLM-Empowered Style Attacks\dataset\phish_sample_30k"

    # 处理数据集，最多处理10个
    extracted_data = process_dataset(dataset_folder, max_items=100)

    # 保存结果
    save_results(extracted_data, r"E:\1study\papers\Cyber Security\Fake News in Sheep’s Clothing——Robust Fake News "
                                 r"Detection Against LLM-Empowered Style Attacks\dataset\phish_sample_30k.json")
    # 打印统计信息
    print(f"\n处理完成！共成功处理了 {len(extracted_data)} 个网站数据")

