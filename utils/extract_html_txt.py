#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:extract_html_txt.py
# author:ZCJ
# datetime:2025-07-21 19:49
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import os
from bs4 import BeautifulSoup
import chardet
import pickle
from openai import OpenAI

client = OpenAI(api_key="sk-xmmzrubysuhvchvqblwisirqmtlsicrkgrhzxjslgvgpzfde",
                base_url="https://api.siliconflow.cn/v1")

MODEL_NAME = "deepseek-ai/DeepSeek-V3"

def process_html_file(file_path):
    # 使用chardet检测文件编码
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

    # 使用检测到的编码读取文件
    with open(file_path, 'r', encoding=encoding) as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(strip=True)
    return text


def batch_process_folders(folder_paths, output_dir):
    for index, folder_path in enumerate(folder_paths):
        html_file_path = os.path.join(folder_path, 'html.txt')
        if os.path.isfile(html_file_path):
            text = process_html_file(html_file_path)

            # 生成测试文件路径
            output_file_name_test = os.path.basename(folder_path) + str(index) + '_test.pkl'
            output_file_path_test = os.path.join(output_dir, 'news_articles', output_file_name_test)

            # 保存测试文本到输出文件
            with open(output_file_path_test, 'wb') as out_f:
                pickle.dump(text, out_f)
            print(f"Processed file: {html_file_path}, Saved to: {output_file_path_test}")

            # 生成训练文件路径
            output_file_name_train = os.path.basename(folder_path) + str(index) + '_train.pkl'
            output_file_path_train = os.path.join(output_dir, 'news_articles', output_file_name_train)

            # 保存训练文本到输出文件
            with open(output_file_path_train, 'wb') as out_f:
                pickle.dump(text, out_f)
            print(f"Processed file: {html_file_path}, Saved to: {output_file_path_train}")

            # 生成emotionally_triggering风格文件
            output_file_name_emotionally_triggering = os.path.basename(folder_path) + str(
                index) + '_train_emotionally_triggering.pkl'
            output_file_path_emotionally_triggering = os.path.join(output_dir, 'reframings',
                                                                   output_file_name_emotionally_triggering)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'user',
                     'content': "Rewrite the following article in an emotionally triggering tone:" + text}
                ],
                temperature=0.7,
                max_tokens=512
            )

            emotionally_triggering_text = response.choices[0].message.content

            # 保存emotionally_triggering风格文件
            with open(output_file_path_emotionally_triggering, 'wb') as out_f:
                pickle.dump(emotionally_triggering_text, out_f)
                print(index, output_file_name_emotionally_triggering)

            # 生成neutral风格文件
            output_file_name_neutral = os.path.basename(folder_path) + str(index) + '_train_neutral.pkl'
            output_file_path_neutral = os.path.join(output_dir, 'reframings', output_file_name_neutral)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'user',
                     'content': "Rewrite the following article in a neutral tone:" + text}
                ],
                temperature=0.7,
                max_tokens=512
            )
            neutral_text = response.choices[0].message.content

            # 保存neutral风格文件
            with open(output_file_path_neutral, 'wb') as out_f:
                pickle.dump(neutral_text, out_f)
                print(index, output_file_name_neutral)

            # 生成objective风格文件
            output_file_name_objective = os.path.basename(folder_path) + str(index) + '_train_objective.pkl'
            output_file_path_objective = os.path.join(output_dir, 'reframings', output_file_name_objective)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'user',
                     'content': "Rewrite the following article in an objective tone:" + text}
                ],
                temperature=0.7,
                max_tokens=512
            )
            objective_text = response.choices[0].message.content

            # 保存objective风格文件
            with open(output_file_path_objective, 'wb') as out_f:
                pickle.dump(objective_text, out_f)
                print(index, output_file_name_objective)

            # 生成sensational风格文件
            output_file_name_sensational = os.path.basename(folder_path) + str(index) + '_train_sensational.pkl'
            output_file_path_sensational = os.path.join(output_dir, 'reframings', output_file_name_sensational)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'user',
                     'content': "Rewrite the following article in a sensational tone:" + text}
                ],
                temperature=0.7,
                max_tokens=512
            )
            sensational_text = response.choices[0].message.content
            # 保存sensational风格文件
            with open(output_file_path_sensational, 'wb') as out_f:
                pickle.dump(sensational_text, out_f)
                print(index, output_file_name_sensational)

            # veracity_attributions

            # 生成fake_standard_neutral_sensational文件
            output_file_name_neutral_sensational = os.path.basename(folder_path) + str(index) + '_fake_standards_neutral_sensational.pkl'
            output_file_path_neutral_sensational = os.path.join(output_dir, 'veracity_attributions', output_file_name_neutral_sensational)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'user',
                     'content': "Article1:" + neutral_text + "\n" +
                                "Article2:" + sensational_text + "\n" +
                                "Question:Which ofthe following problems does the articles have?Lack of credible sources, False or "
                                "misleading information, Biasedopinion, Inconsistencies with reputable sources. If multiple "
                                "op-tions apply, provide a comma-separated list ordered from most toleast related. Answer “No "
                                "problems” if none of the options apply."}
                ],
                temperature=0,
                max_tokens=512
            )

            neutral_sensational = response.choices[0].message.content

            # 保存fake_standard_neutral_sensational文件
            with open(output_file_path_neutral_sensational, 'wb') as out_f:
                pickle.dump(neutral_sensational, out_f)
                print(index, output_file_path_neutral_sensational)

            # 生成fake_standards_objective_emotionally_triggering文件
            output_file_name_objective_emotionally_triggering = os.path.basename(folder_path) + str(index) + '_fake_standards_objective_emotionally_triggering.pkl'
            output_file_path_objective_emotionally_triggering = os.path.join(output_dir, 'veracity_attributions', output_file_name_objective_emotionally_triggering)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'user',
                     'content': "Article1:" + objective_text + "\n" +
                                "Article2:" + emotionally_triggering_text + "\n" +
                                "Question:Which ofthe following problems does the articles have?Lack of credible sources, False or "
                                "misleading information, Biasedopinion, Inconsistencies with reputable sources. If multiple "
                                "op-tions apply, provide a comma-separated list ordered from most toleast related. Answer “No "
                                "problems” if none of the options apply."
                    }
                ]
            )

            objective_emotionally_triggering = response.choices[0].message.content

            # 保存fake_standards_objective_emotionally_triggering文件
            with open(output_file_path_objective_emotionally_triggering, 'wb') as out_f:
                pickle.dump(objective_emotionally_triggering, out_f)
                print(index, output_file_path_objective_emotionally_triggering)

            # Attack Formulation
            # 使用CNN风格改写文章
            output_file_name_A = os.path.basename(folder_path) + str(index) + '_test_adv_A.pkl'
            output_file_path_A = os.path.join(output_dir, 'adversarial_test', output_file_name_A)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'user',
                     'content': "Rewrite the following article using the style of CNN:" + text}
                ],
                temperature=0.7,
                max_tokens=512
            )
            A = response.choices[0].message.content
            with open(output_file_path_A, 'wb') as out_f:
                pickle.dump(A, out_f)
                print(index, output_file_name_A)

            # 使用The New York Times风格改写文章
            output_file_name_B = os.path.basename(folder_path) + str(index) + '_test_adv_B.pkl'
            output_file_path_B = os.path.join(output_dir, 'adversarial_test', output_file_name_B)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'user',
                     'content': "Rewrite the following article using the style of The New York Times:" + text}
                ],
                temperature=0.7,
                max_tokens=512
            )
            B = response.choices[0].message.content
            with open(output_file_path_B, 'wb') as out_f:
                pickle.dump(B, out_f)
                print(index, output_file_name_B)

            # 使用National Enquirer风格改写文章
            output_file_name_C = os.path.basename(folder_path) + str(index) + '_test_adv_C.pkl'
            output_file_path_C = os.path.join(output_dir, 'adversarial_test', output_file_name_C)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'user',
                     'content': "Rewrite the following article using the style of National Enquirer:" + text}
                ],
                temperature=0.7,
                max_tokens=512
            )
            C = response.choices[0].message.content
            with open(output_file_path_C, 'wb') as out_f:
                pickle.dump(C, out_f)
                print(index, output_file_name_C)

            # 使用The Sun风格改写文章
            output_file_name_D = os.path.basename(folder_path) + str(index) + '_test_adv_D.pkl'
            output_file_path_D = os.path.join(output_dir, 'adversarial_test', output_file_name_D)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'user',
                     'content': "Rewrite the following article using the style of The Sun:" + text}
                ],
                temperature=0.7,
                max_tokens=512
            )
            D = response.choices[0].message.content
            with open(output_file_path_D, 'wb') as out_f:
                pickle.dump(D, out_f)
                print(index, output_file_name_D)


if __name__ == "__main__":
    folder_paths = []
    for root, dirs, files in os.walk('../data_test'):
        if 'html.txt' in files:
            folder_paths.append(root)
    output_directory = '../data1'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    batch_process_folders(folder_paths, output_directory)
