"""
@Time:
@File: extract_html_txt2.py
@Software: PyCharm
@Description: 将提取出来的文本进行LLM改写
"""
"""
@Time:
@File: extract_html_txt2.py
@Software: PyCharm
@Description: 将提取出来的文本进行LLM改写
"""
import os
import pickle
import numpy as np
from zai import ZhipuAiClient
from tqdm import tqdm
import concurrent.futures


# 假设新API支持设置类似temperature和max_tokens的参数，这里自定义设置方式
def set_custom_params(params, temperature, max_tokens):
    params['temperature'] = temperature
    params['max_tokens'] = max_tokens
    return params


client = ZhipuAiClient(api_key="dc37977a93224831851fa916ee99fd65.vzXMwoT2Ow0sT0O9")
MODEL_NAME = "glm-4-flash-250414"


def process_single_news(text, style, temperature, max_tokens):
    messages = [
        {'role': 'user',
         'content': f"Rewrite the following article in a {style} tone:" + text}
    ]
    custom_params = {'model': MODEL_NAME,'messages': messages}
    custom_params = set_custom_params(custom_params, temperature, max_tokens)
    response = client.chat.completions.create(**custom_params)
    return response.choices[0].message.content


def process_news(news_list, styles, temperature = 0.7, max_tokens = 512):
    all_processed_news = {style: [] for style in styles}
    with concurrent.futures.ThreadPoolExecutor(max_workers = 30) as executor:
        futures = []
        for style in styles:
            for text in news_list:
                future = executor.submit(process_single_news, text, style, temperature, max_tokens)
                futures.append((future, style))
        future_style_dict = {f: s for f, s in futures}
        for future in tqdm(concurrent.futures.as_completed([f[0] for f in futures]), total = len(futures),
                           desc = "Processing news in all styles"):
            try:
                processed_text = future.result()
                style = future_style_dict[future]
                all_processed_news[style].append(processed_text)
            except Exception as e:
                print(f"Error processing news for style {style}: {e}")
    return all_processed_news


def read_and_process_pkl(pkl_path, output_dir):
    if not os.path.exists(pkl_path):
        print(f"The file {pkl_path} does not exist.")
        return

    pkl_path = os.path.join(output_dir, 'news_articles', 'news_train.pkl')
    if not os.path.exists(pkl_path):
        print(f"The file {pkl_path} does not exist.")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    news_list = data.get('news', [])
    labels = data.get('labels', [])

    styles = ['neutral', 'objective','sensational', 'emotionally_triggering']
    all_processed_news = process_news(news_list, styles)

    for style, processed_news in all_processed_news.items():
        pkl_file_path = os.path.join(output_dir,'reframings', f'news_train_{style}.pkl')
        with open(pkl_file_path, 'wb') as f:
            pickle.dump({'rewritten': processed_news, 'labels': labels}, f)


def process_attack_formulation_read_and_process_pkl(pkl_path, output_dir):
    if not os.path.exists(pkl_path):
        print(f"The file {pkl_path} does not exist.")
        return

    pkl_path = os.path.join(output_dir, 'news_articles', 'news_test.pkl')
    if not os.path.exists(pkl_path):
        print(f"The file {pkl_path} does not exist.")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    news_list = data.get('news', [])
    labels = data.get('labels', [])

    # Attack Formulation 'adversarial_test'
    label_0_media_styles = ['National Enquirer', 'The Sun']
    label_1_media_styles = ['CNN', 'The New York Times']
    letter_mapping = {i: chr(65 + i) for i in range(len(label_0_media_styles) + len(label_1_media_styles))}

    label_0_news = [news for news, label in zip(news_list, labels) if label == 0]
    label_1_news = [news for news, label in zip(news_list, labels) if label == 1]

    for i, media_style in enumerate(label_0_media_styles):
        processed_news = process_attack_formulation(label_0_news, media_style)
        pkl_file_path = os.path.join(output_dir, 'adversarial_test',
                                     f'news_test_adv_{letter_mapping[i]}.pkl')
        with open(pkl_file_path, 'wb') as f:
            relevant_labels = [label for label in labels if label == 0]
            pickle.dump({'news': processed_news, 'labels': relevant_labels}, f)

    for i, media_style in enumerate(label_1_media_styles):
        processed_news = process_attack_formulation(label_1_news, media_style)
        pkl_file_path = os.path.join(output_dir, 'adversarial_test',
                                     f'news_test_adv_{letter_mapping[len(label_0_media_styles) + i]}.pkl')
        with open(pkl_file_path, 'wb') as f:
            relevant_labels = [label for label in labels if label == 1]
            pickle.dump({'news': processed_news, 'labels': relevant_labels}, f)


def process_single_attack_formulation(text, media_style, temperature, max_tokens):
    messages = [
        {'role': 'user',
         'content': f"Rewrite the following article using the style of {media_style}:" + text}
    ]
    custom_params = {'model': MODEL_NAME,'messages': messages}
    custom_params = set_custom_params(custom_params, temperature, max_tokens)
    response = client.chat.completions.create(**custom_params)
    return response.choices[0].message.content


def process_attack_formulation(news_list, media_style, temperature = 0.7, max_tokens = 512):
    processed_news = []
    with concurrent.futures.ThreadPoolExecutor(max_workers = 30) as executor:
        future_to_text = {
            executor.submit(process_single_attack_formulation, text, media_style, temperature, max_tokens): text
            for text in news_list}
        for future in tqdm(concurrent.futures.as_completed(future_to_text), total = len(news_list),
                           desc = f"Processing {media_style} style"):
            try:
                processed_text = future.result()
                processed_news.append(processed_text)
            except Exception as e:
                print(f"Error processing attack formulation for {media_style}: {e}")
    return processed_news


def get_veracity_labels(text):
    messages = [
        {"role": "user",
         "content": "Please analyze the following article and return a list with four elements. "
                    "Each element corresponds to one of the following problems: lack of credible sources, "
                    "false or misleading information, biased views, and inconsistency with authoritative sources. "
                    "If the article has the corresponding problem, set the element to 1; otherwise, set it to 0. "
                    "For example, if the article only has a problem of lack of credible sources, return [1, 0, 0, 0]. "
                    "Article: " + text},
        {"role": "assistant",
         "content": "Please tell me which option you would like me to choose."},
        {"role": "user", "content": "Please just return the list as requested."}
    ]
    custom_params = {
       'model': MODEL_NAME,
      'messages': messages,
        'thinking': {
            "type": "disabled"
        },
       'stream': True,
       'max_tokens': 500,
        'temperature': 0.7
    }
    response = client.chat.completions.create(**custom_params)
    result = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            result += chunk.choices[0].delta.content
    try:
        if result.startswith('[') and result.endswith(']'):
            return np.array([float(i) for i in result.strip('[]').split(',')])
        else:
            return np.array([0., 0., 0., 0.])
    except ValueError:
        return np.array([0., 0., 0., 0.])


def generate_veracity_pkl(output_dir, mainstream_style, tabloid_style):
    mainstream_file = os.path.join(output_dir,'reframings', f'news_train_{mainstream_style}.pkl')
    tabloid_file = os.path.join(output_dir,'reframings', f'news_train_{tabloid_style}.pkl')

    with open(mainstream_file, 'rb') as f:
        mainstream_data = pickle.load(f)
    with open(tabloid_file, 'rb') as f:
        tabloid_data = pickle.load(f)

    # 从原始pkl文件读取的数据中获取news_list
    pkl_path = os.path.join(output_dir, 'news_articles', 'news_train.pkl')
    with open(pkl_path, 'rb') as f:
        original_data = pickle.load(f)
    news_list = original_data.get('news', [])

    mainstream_texts = mainstream_data['rewritten']
    tabloid_texts = tabloid_data['rewritten']

    orig_fg = []
    mainstream_fg = []
    tabloid_fg = []

    for text in tqdm(news_list, desc=f"Processing original news for orig_fg"):
        label = get_veracity_labels(text)
        orig_fg.append(label)
    orig_fg = np.array(orig_fg)

    for text in tqdm(mainstream_texts, desc=f"Processing {mainstream_style} for mainstream_fg"):
        label = get_veracity_labels(text)
        mainstream_fg.append(label)
    mainstream_fg = np.array(mainstream_fg)

    for text in tqdm(tabloid_texts, desc=f"Processing {tabloid_style} for tabloid_fg"):
        label = get_veracity_labels(text)
        tabloid_fg.append(label)
    tabloid_fg = np.array(tabloid_fg)

    data_dict = {
        'orig_fg': orig_fg,
      'mainstream_fg': mainstream_fg,
        'tabloid_fg': tabloid_fg,
        'classes': ['Lack of credible sources', 'False or misleading information', 'Biased opinion', 'Inconsistencies with reputable sources']
    }

    pkl_file_path = os.path.join(output_dir,'veracity_attributions', f'news_fake_standards_{mainstream_style}_{tabloid_style}.pkl')
    if not os.path.exists(os.path.dirname(pkl_file_path)):
        os.makedirs(os.path.dirname(pkl_file_path))
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    pkl_path = 'data.pkl'
    output_directory = '../data'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    read_and_process_pkl(pkl_path, output_directory)
    process_attack_formulation_read_and_process_pkl(pkl_path, output_directory)

    generate_veracity_pkl(output_directory, 'neutral','sensational')
    generate_veracity_pkl(output_directory, 'objective', 'emotionally_triggering')


