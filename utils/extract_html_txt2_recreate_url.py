import os
import pickle
import numpy as np
from zai import ZhipuAiClient
from tqdm import tqdm
import concurrent.futures
import re


# 初始化智谱AI客户端
client = ZhipuAiClient(api_key="dc37977a93224831851fa916ee99fd65.vzXMwoT2Ow0sT0O9")
MODEL_NAME = "glm-4-flash-250414"


def set_custom_params(params, temperature, max_tokens):
    """设置模型参数"""
    params['temperature'] = temperature
    params['max_tokens'] = max_tokens
    return params


def clean_web_text(text):
    """清洗网页文本，移除多余空格和特殊字符"""
    text = re.sub(r'\s+', ' ', text).strip()  # 合并空白字符
    text = re.sub(r'[^\w\s.,!?]', '', text)   # 移除特殊字符
    return text


def process_single_webpage(text, style, temperature=0.7, max_tokens=512):
    """处理单个网页文本，按指定风格改写"""
    # 清洗文本
    cleaned_text = clean_web_text(text)

    # 根据风格生成不同的改写提示
    style_prompts = {
        'standard': "Rewrite the following webpage content in a standard, trustworthy style. "
                   "Use formal language, clear structure, and avoid exaggerated claims: ",
        'deceptive': "Rewrite the following webpage content to make it deceptive and phishing-like. "
                    "Use urgent language, suspicious requests, and misleading information: ",
        'neutral': "Rewrite the following webpage content in a neutral, objective style "
                  "without emotional language or persuasive tactics: ",
        'alarming': "Rewrite the following webpage content to create a sense of alarm and urgency "
                   "to prompt immediate action from the reader: "
    }

    messages = [
        {'role': 'user',
         'content': f"{style_prompts.get(style, style_prompts['neutral'])}{cleaned_text}"}
    ]

    custom_params = {'model': MODEL_NAME, 'messages': messages}
    custom_params = set_custom_params(custom_params, temperature, max_tokens)

    try:
        response = client.chat.completions.create(**custom_params)
        return response.choices[0].message.content
    except Exception as e:
        print(f"风格改写失败: {str(e)}")
        return cleaned_text  # 失败时返回原始清洗文本


def process_webpages(web_list, styles, temperature=0.7, max_tokens=512):
    """批量处理网页文本，生成多种风格的改写版本"""
    all_processed = {style: [] for style in styles}

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:  # 降低并发数避免API限制
        futures = []
        for style in styles:
            for text in web_list:
                future = executor.submit(
                    process_single_webpage,
                    text, style, temperature, max_tokens
                )
                futures.append((future, style))

        future_style_map = {f: s for f, s in futures}

        for future in tqdm(
            concurrent.futures.as_completed([f[0] for f in futures]),
            total=len(futures),
            desc="处理网页文本风格改写"
        ):
            try:
                processed_text = future.result()
                style = future_style_map[future]
                all_processed[style].append(processed_text)
            except Exception as e:
                print(f"处理失败: {str(e)}")

    return all_processed


def read_and_process_webpages(pkl_path, output_dir):
    """读取网页数据集并生成风格改写版本"""
    # 构建正确的文件路径
    input_path = os.path.join(output_dir, 'web_articles', 'web_train.pkl')
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return

    # 加载原始数据
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    web_list = data.get('news', [])  # 网页文本内容
    urls = data.get('url', [])       # 保留URL信息
    labels = data.get('labels', [])  # 标签（0=正常，1=钓鱼）

    # 定义适合钓鱼网站检测的风格
    styles = ['neutral', 'objective','sensational', 'emotionally_triggering']
    all_processed = process_webpages(web_list, styles)

    # 保存处理结果
    output_subdir = os.path.join(output_dir, 'reframings')
    os.makedirs(output_subdir, exist_ok=True)

    for style, processed in all_processed.items():
        output_path = os.path.join(output_subdir, f'web_train_{style}.pkl')
        with open(output_path, 'wb') as f:
            # 保存改写文本、原始URL和标签
            pickle.dump({
                'rewritten': processed,
                'alternate_rewritten': [clean_web_text(t) for t in web_list],  # 备用版本
                'url': urls,
                'labels': labels
            }, f)
        print(f"已保存 {style} 风格数据至 {output_path}")


def process_attack_formulation_read_and_process_pkl(pkl_path, output_dir):
    """根据标签分类生成对抗性测试集"""
    # 验证输入文件是否存在
    # if not os.path.exists(pkl_path):
    #     print(f"文件不存在: {pkl_path}")
    #     return

    # 构建网页数据集路径（网页测试集）
    web_test_path = os.path.join(output_dir, 'web_articles', 'web_test.pkl')
    if not os.path.exists(web_test_path):
        print(f"网页测试集文件不存在: {web_test_path}")
        return

    # 加载网页测试数据
    with open(web_test_path, 'rb') as f:
        data = pickle.load(f)

    web_list = data.get('news', [])  # 网页文本内容
    urls = data.get('url', [])  # 网页URL
    labels = data.get('labels', [])  # 标签（0=正常网页，1=钓鱼网页）

    # 定义对抗性风格（按标签分组设置不同风格）
    # 针对正常网页(0)的对抗性风格：使其更像钓鱼网页
    label_0_attack_styles = [
        'highly_urgent_scam',  # 高度紧急的诈骗风格
        'fake_financial_service'  # 伪造金融服务风格
    ]

    # 针对钓鱼网页(1)的对抗性风格：使其更像正常网页
    label_1_attack_styles = [
        'legitimate_business',  # 合法企业风格
        'official_government_service'  # 官方政府服务风格
    ]

    # 创建字母映射（A、B、C、D...）
    total_styles = len(label_0_attack_styles) + len(label_1_attack_styles)
    letter_mapping = {i: chr(65 + i) for i in range(total_styles)}

    # 按标签分组网页文本
    label_0_web = [web for web, label in zip(web_list, labels) if label == 0]
    label_1_web = [web for web, label in zip(web_list, labels) if label == 1]
    label_0_urls = [url for url, label in zip(urls, labels) if label == 0]
    label_1_urls = [url for url, label in zip(urls, labels) if label == 1]

    # 处理正常网页(0)：生成钓鱼风格的对抗样本
    for i, attack_style in enumerate(label_0_attack_styles):
        # 为每种风格定义针对性提示词
        if attack_style == 'highly_urgent_scam':
            prompt = "Rewrite this webpage to create a highly urgent scam: " \
                     "Use alarming language, claim account suspension, " \
                     "and demand immediate personal information verification: "
        else:  # fake_financial_service
            prompt = "Rewrite this webpage to impersonate a financial service: " \
                     "Include fake account statements, payment requests, " \
                     "and banking terminology to deceive users: "

        # 生成对抗性文本
        processed_web = process_attack_formulation(label_0_web, prompt)

        # 保存结果
        output_file = os.path.join(output_dir, 'adversarial_test',
                                   f'web_test_adv_{letter_mapping[i]}.pkl')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            # 保留原始标签(0)和对应的URL
            pickle.dump({
                'news': processed_web,
                'url': label_0_urls,
                'labels': [0] * len(processed_web)
            }, f)
        print(f"已保存正常网页对抗样本 {letter_mapping[i]} 至 {output_file}")

    # 处理钓鱼网页(1)：生成正常风格的对抗样本
    for i, attack_style in enumerate(label_1_attack_styles):
        # 计算字母映射索引（跳过前一组的索引）
        index = len(label_0_attack_styles) + i

        # 为每种风格定义针对性提示词
        if attack_style == 'legitimate_business':
            prompt = "Rewrite this webpage to look like a legitimate business: " \
                     "Use professional language, clear contact information, " \
                     "and transparent service descriptions: "
        else:  # official_government_service
            prompt = "Rewrite this webpage to look like an official government service: " \
                     "Use formal language, clear process explanations, " \
                     "and official-sounding instructions without deception: "

        # 生成对抗性文本
        processed_web = process_attack_formulation(label_1_web, prompt)

        # 保存结果
        output_file = os.path.join(output_dir, 'adversarial_test',
                                   f'web_test_adv_{letter_mapping[index]}.pkl')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            # 保留原始标签(1)和对应的URL
            pickle.dump({
                'news': processed_web,
                'url': label_1_urls,
                'labels': [1] * len(processed_web)
            }, f)
        print(f"已保存钓鱼网页对抗样本 {letter_mapping[index]} 至 {output_file}")


def process_single_attack(text, prompt, temperature=0.8, max_tokens=512):
    """生成单个对抗性样本"""
    cleaned_text = clean_web_text(text)
    messages = [{'role': 'user', 'content': f"{prompt}{cleaned_text}"}]

    custom_params = {'model': MODEL_NAME, 'messages': messages}
    custom_params = set_custom_params(custom_params, temperature, max_tokens)

    try:
        response = client.chat.completions.create(**custom_params)
        return response.choices[0].message.content
    except Exception as e:
        print(f"对抗样本生成失败: {str(e)}")
        return cleaned_text


def process_attack_formulation(web_list, prompt, temperature=0.8, max_tokens=512):
    """批量生成对抗性样本"""
    processed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_single_attack, text, prompt, temperature, max_tokens): text
            for text in web_list
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(web_list),
            desc="生成对抗性样本"
        ):
            try:
                processed_text = future.result()
                processed.append(processed_text)
            except Exception as e:
                print(f"处理失败: {str(e)}")

    return processed


def get_phishing_features(text):
    """提取钓鱼网站特征标签（4维细粒度特征）"""
    cleaned_text = clean_web_text(text)
    messages = [
        {"role": "user",
         "content": "Analyze the following webpage content and return a list of 4 elements. "
                    "Each element indicates the presence (1) or absence (0) of a phishing characteristic:\n"
                    "1. Urgent requests for personal information or actions\n"
                    "2. Poor grammar, spelling errors, or unprofessional language\n"
                    "3. Suspicious links, fake URLs, or misleading domain names\n"
                    "4. Fake logos, brand impersonation, or false authority claims\n"
                    "Return only the list of 0s and 1s, e.g., [1,0,1,0]\n"
                    f"Webpage content: {cleaned_text}"},
        {"role": "assistant", "content": "I will return only the list as requested."}
    ]

    custom_params = {
        'model': MODEL_NAME,
        'messages': messages,
        'thinking': {"type": "disabled"},
        'stream': True,
        'max_tokens': 100,
        'temperature': 0.3  # 降低随机性，确保判断更稳定
    }

    try:
        response = client.chat.completions.create(**custom_params)
        result = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content

        # 解析结果
        if result.startswith('[') and result.endswith(']'):
            return np.array([float(i.strip()) for i in result.strip('[]').split(',')])
        return np.array([0, 0, 0, 0])
    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return np.array([0, 0, 0, 0])


def generate_phishing_features(output_dir, style_pairs):
    """生成钓鱼网站细粒度特征标签（4维）"""
    for mainstream, deceptive in style_pairs:
        # 加载两种风格的文本数据
        mainstream_path = os.path.join(output_dir, 'reframings', f'web_train_{mainstream}.pkl')
        deceptive_path = os.path.join(output_dir, 'reframings', f'web_train_{deceptive}.pkl')

        if not os.path.exists(mainstream_path) or not os.path.exists(deceptive_path):
            print(f"风格数据文件缺失: {mainstream_path} 或 {deceptive_path}")
            continue

        with open(mainstream_path, 'rb') as f:
            mainstream_data = pickle.load(f)
        with open(deceptive_path, 'rb') as f:
            deceptive_data = pickle.load(f)

        # 加载原始网页文本
        orig_path = os.path.join(output_dir, 'web_articles', 'web_train.pkl')
        with open(orig_path, 'rb') as f:
            orig_data = pickle.load(f)
        orig_texts = orig_data.get('news', [])

        # 提取细粒度特征
        orig_fg = []
        mainstream_fg = []
        deceptive_fg = []

        for text in tqdm(orig_texts, desc="提取原始文本特征"):
            orig_fg.append(get_phishing_features(text))

        for text in tqdm(mainstream_data['rewritten'], desc=f"提取{mainstream}风格特征"):
            mainstream_fg.append(get_phishing_features(text))

        for text in tqdm(deceptive_data['rewritten'], desc=f"提取{deceptive}风格特征"):
            deceptive_fg.append(get_phishing_features(text))

        # 保存特征数据
        output_subdir = os.path.join(output_dir, 'veracity_attributions')
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, f'web_fake_standards_{mainstream}_{deceptive}.pkl')

        with open(output_path, 'wb') as f:
            pickle.dump({
                'orig_fg': np.array(orig_fg),
                'standard_fg': np.array(mainstream_fg),  # 标准风格特征
                'deceptive_fg': np.array(deceptive_fg),  # 欺骗风格特征
                'classes': [
                    'Urgent requests for info/actions',
                    'Grammar/spelling errors',
                    'Suspicious links/URLs',
                    'Fake logos/impersonation'
                ]
            }, f)
        print(f"已保存特征数据至 {output_path}")


if __name__ == "__main__":
    output_directory = '../data'
    os.makedirs(output_directory, exist_ok=True)

    # 1. 生成风格改写的训练数据
    read_and_process_webpages('', output_directory)

    # 2. 生成对抗性测试数据
    process_attack_formulation_read_and_process_pkl('', output_directory)

    # 3. 生成细粒度特征标签（针对不同风格组合）
    generate_phishing_features(
        output_directory,
        style_pairs=[('neutral', 'sensational'), ('objective', 'emotionally_triggering')]
    )
