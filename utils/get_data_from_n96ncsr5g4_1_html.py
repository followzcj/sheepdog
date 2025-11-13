#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:get_data_from_n96ncsr5g4_1_html.py
# author:ZCJ
# datetime:2025-10-20 18:41
# software: PyCharm

"""
this is function  description 
"""

# import module your need


import os
import json
import pymysql
from pymysql.cursors import DictCursor
from collections import defaultdict


def read_html_source(file_path):
    """从TXT文件中读取原始HTML内容"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # 直接读取原始HTML内容，不做任何解析提取
            html_source = f.read()
        return html_source
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return None
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None


def build_file_index(root_folder):
    """
    构建文件索引，记录所有TXT文件的路径
    键: 文件名(不含路径)，值: 文件完整路径
    """
    file_index = defaultdict(list)

    # 递归遍历所有子文件夹
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.txt'):
                # 存储文件名与完整路径的映射
                file_index[filename].append(os.path.join(dirpath, filename))

    return file_index


def get_sql_data(host, user, password, db_name):
    """从MySQL数据库的index表读取数据"""
    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db_name,
            charset='utf8mb4',
            cursorclass=DictCursor
        )

        with connection:
            with connection.cursor() as cursor:
                sql = "SELECT `rec_id`, `url`, `website`, `result` FROM `index`"
                cursor.execute(sql)
                return cursor.fetchall()

    except pymysql.MySQLError as e:
        print(f"数据库操作错误: {str(e)}")
        return None


def process_sql_and_files(sql_data, root_folder, max_items=1000):
    """结合SQL数据和多文件夹中的TXT文件提取信息"""
    results = []
    count = 0

    if not sql_data:
        print("没有获取到有效的SQL数据，无法继续处理")
        return results

    # 预构建文件索引，提高查找效率
    print(f"正在扫描 {root_folder} 下的所有TXT文件...")
    file_index = build_file_index(root_folder)
    print(f"扫描完成，共发现 {sum(len(v) for v in file_index.values())} 个TXT文件")

    for item in sql_data:
        # if count >= max_items:
        #     break

        rec_id = item['rec_id']
        url = item['url']
        website = item['website']
        result = item['result']

        # 生成目标TXT文件名
        base_name = os.path.splitext(website)[0]
        target_filename = f"{base_name}.txt"

        # 在索引中查找文件
        if target_filename not in file_index:
            print(f"跳过记录 {rec_id}: 未找到 {target_filename}")
            continue

        # 处理找到的文件（如果有多个同名文件，只取第一个）
        file_path = file_index[target_filename][0]
        if len(file_index[target_filename]) > 1:
            print(f"警告: 发现多个 {target_filename}，将使用第一个: {file_path}")

        # 读取原始HTML内容（不再提取文本）
        html_source = read_html_source(file_path)
        if html_source is None:
            print(f"跳过记录 {rec_id}: 无法读取文件内容")
            continue

        # 添加到结果，字段名从html_text改为html_source更贴切
        results.append({
            'url': url,
            'html_text': html_source,  # 保存原始HTML代码
            'label': result,
        })

        count += 1
        if count % 10 == 0:
            print(f"已处理 {count} 条记录")

    return results


def save_results(results, output_file):
    """将提取的结果保存为JSON文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n处理完成！共提取 {len(results)} 条有效记录")
        print(f"结果已保存至: {output_file}")
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")


if __name__ == "__main__":
    # 数据库配置
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'root',
        'password': '202826',
        'db_name': 'phish_webs'
    }

    # 文件路径配置（根目录，脚本会自动搜索其子文件夹）
    ROOT_FOLDER = r"E:\1study\papers\Cyber_Security\sheepdog\dataset\n96ncsr5g4-1\dataset"  # 所有TXT文件的根目录
    OUTPUT_FILE = r"E:\1study\papers\Cyber_Security\sheepdog\dataset\extracted_sql_txt_results_html.json"

    # 最大处理记录数
    MAX_ITEMS = 1000

    print("开始从数据库读取数据...")
    sql_data = get_sql_data(
        DB_CONFIG['host'],
        DB_CONFIG['user'],
        DB_CONFIG['password'],
        DB_CONFIG['db_name']
    )

    if sql_data:
        print(f"成功读取 {len(sql_data)} 条记录，开始处理文件...")
        extracted_data = process_sql_and_files(sql_data, ROOT_FOLDER, MAX_ITEMS)
        save_results(extracted_data, OUTPUT_FILE)
    else:
        print("无法获取数据，程序终止")