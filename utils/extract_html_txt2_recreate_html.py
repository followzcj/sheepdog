import os
import pickle
import numpy as np
import re
from zhipuai import ZhipuAI
from tqdm import tqdm
import concurrent.futures

# 初始化智谱AI客户端
client = ZhipuAI(api_key="dc37977a93224831851fa916ee99fd65.vzXMwoT2Ow0sT0O9")
MODEL_NAME = "glm-4-flash"


def set_custom_params(params, temperature, max_tokens):
    """设置模型参数"""
    params['temperature'] = temperature
    params['max_tokens'] = max_tokens
    return params


def clean_html(html):
    """轻度清洗HTML，保留标签结构"""
    # 只去除多余空白，保留标签结构
    html = re.sub(r'\n\s+', '\n', html).strip()
    return html


def process_single_html(html, style, temperature=0.7, max_tokens=2048):
    """处理单个HTML，同时改写文本内容和标签样式"""
    cleaned_html = clean_html(html)

    # 增强的风格提示，包含HTML结构改写要求
    style_prompts = {
        'standard': "Rewrite the following HTML content to make it look standard and trustworthy. "
                    "For text content: Use formal language, clear structure, avoid exaggerated claims. "
                    "For HTML tags/styles: Use clean, professional formatting, standard fonts, "
                    "appropriate spacing, and official-looking layouts. "
                    "Preserve all functional elements but make them look legitimate. "
                    "Return complete HTML code without explanations:\n",

        'deceptive': "Rewrite the following HTML content to make it deceptive and phishing-like. "
                     "For text content: Use urgent language, suspicious requests, misleading information. "
                     "For HTML tags/styles: Use flashy colors, excessive bold/underline, fake security badges, "
                     "cluttered layout with urgent action buttons. "
                     "Return complete HTML code without explanations:\n",

        'neutral': "Rewrite the following HTML content in a neutral, objective style. "
                   "For text content: Avoid emotional language or persuasive tactics. "
                   "For HTML tags/styles: Use simple, unadorned formatting with standard elements. "
                   "Return complete HTML code without explanations:\n",

        'alarming': "Rewrite the following HTML content to create alarm and urgency. "
                    "For text content: Emphasize threats, time sensitivity, and immediate action needs. "
                    "For HTML tags/styles: Use red warnings, flashing elements (where appropriate), "
                    "large urgent buttons, and attention-grabbing formatting. "
                    "Return complete HTML code without explanations:\n",

        'legitimate_business': "Rewrite the following HTML to look like a legitimate business website. "
                               "For text content: Use professional language, clear contact info, transparent services. "
                               "For HTML tags/styles: Use corporate colors, professional fonts, "
                               "trust indicators (address, phone, privacy policy links), clean navigation. "
                               "Return complete HTML code without explanations:\n",

        'official_government': "Rewrite the following HTML to look like an official government service. "
                               "For text content: Use formal, authoritative language with clear instructions. "
                               "For HTML tags/styles: Use official-looking layouts, government-style fonts, "
                               "official seals/watermarks (if applicable), and structured information presentation. "
                               "Return complete HTML code without explanations:\n",

        'highly_urgent_scam': "Rewrite the following HTML to look like a highly urgent scam. "
                              "For text content: Claim account suspension, demand immediate personal info verification. "
                              "For HTML tags/styles: Use urgent red alerts, countdown timers, "
                              "fake lock icons, and prominent warning banners. "
                              "Return complete HTML code without explanations:\n",

        'fake_financial': "Rewrite the following HTML to impersonate a financial service. "
                          "For text content: Include fake account statements and payment requests. "
                          "For HTML tags/styles: Use banking-style layouts, fake security badges, "
                          "transaction tables, and official-looking forms. "
                          "Return complete HTML code without explanations:\n"
    }

    # 获取对应风格的提示词
    prompt = style_prompts.get(style, style_prompts['neutral'])

    messages = [
        {'role': 'user', 'content': f"{prompt}{cleaned_html}"}
    ]

    custom_params = {'model': MODEL_NAME, 'messages': messages}
    custom_params = set_custom_params(custom_params, temperature, max_tokens)

    try:
        response = client.chat.completions.create(**custom_params)
        rewritten_html = response.choices[0].message.content
        # 提取HTML内容（去除可能的解释文本）
        if '<html' in rewritten_html:
            rewritten_html = rewritten_html.split('<html', 1)[1]
            rewritten_html = '<html' + rewritten_html
        if '</html>' in rewritten_html:
            rewritten_html = rewritten_html.split('</html>', 1)[0] + '</html>'
        return rewritten_html
    except Exception as e:
        print(f"HTML改写失败: {str(e)}")
        return cleaned_html  # 失败时返回原始清洗后的HTML


def process_htmls(html_list, styles, temperature=0.7, max_tokens=2048):
    """批量处理HTML，生成多种风格的改写版本"""
    all_processed = {style: [] for style in styles}

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:  # 降低并发避免API限制
        futures = []
        for style in styles:
            for html in html_list:
                future = executor.submit(
                    process_single_html,
                    html, style, temperature, max_tokens
                )
                futures.append((future, style))

        future_style_map = {f: s for f, s in futures}

        for future in tqdm(
                concurrent.futures.as_completed([f[0] for f in futures]),
                total=len(futures),
                desc="处理HTML风格改写"
        ):
            try:
                processed_html = future.result()
                style = future_style_map[future]
                all_processed[style].append(processed_html)
            except Exception as e:
                print(f"处理失败: {str(e)}")

    return all_processed


def read_and_process_htmls(pkl_path, output_dir):
    """读取HTML数据集并生成风格改写版本"""
    # 构建正确的文件路径（假设数据中包含html字段）
    input_path = os.path.join(output_dir, 'web_articles', 'web_train.pkl')
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return

    # 加载原始数据
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    html_list = data.get('html', [])  # 假设数据集中有html字段
    urls = data.get('url', [])  # 保留URL信息
    labels = data.get('labels', [])  # 标签（0=正常，1=钓鱼）

    # 定义适合钓鱼网站检测的风格
    styles = ['neutral', 'objective', 'sensational', 'emotionally_triggering']
    all_processed = process_htmls(html_list, styles)

    # 保存处理结果
    output_subdir = os.path.join(output_dir, 'reframings')
    os.makedirs(output_subdir, exist_ok=True)

    for style, processed in all_processed.items():
        output_path = os.path.join(output_subdir, f'web_train_{style}.pkl')
        with open(output_path, 'wb') as f:
            # 保存完整改写后的HTML、原始URL和标签
            pickle.dump({
                'rewritten': processed,
                'alternate_rewritten': [clean_html(html) for html in html_list],  # 备用版本
                'url': urls,
                'labels': labels
            }, f)
        print(f"已保存 {style} 风格HTML至 {output_path}")


def process_attack_formulation_read_and_process_pkl(pkl_path, output_dir):
    """生成对抗性HTML测试集，对所有测试集文本进行多种风格改写"""
    # 构建网页数据集路径
    web_test_path = os.path.join(output_dir, 'web_articles', 'web_test.pkl')
    if not os.path.exists(web_test_path):
        print(f"网页测试集文件不存在: {web_test_path}")
        return

    # 加载网页测试数据
    with open(web_test_path, 'rb') as f:
        data = pickle.load(f)

    html_list = data.get('html', [])  # 所有HTML文本
    urls = data.get('url', [])        # 所有URL
    labels = data.get('labels', [])   # 所有标签（保持不变）

    # 定义所有对抗性风格（按A、B、C、D顺序）
    attack_styles = [
        'highly_urgent_scam',    # A风格：高度紧急的诈骗风格
        'fake_financial',        # B风格：伪造金融服务风格
        'legitimate_business',   # C风格：合法企业风格
        'official_government'    # D风格：官方政府服务风格
    ]

    # 创建字母映射（A、B、C、D...）
    letter_mapping = {i: chr(65 + i) for i in range(len(attack_styles))}

    # 对所有测试集文本按每种风格进行改写
    for i, attack_style in enumerate(attack_styles):
        # 生成对抗性HTML（对所有文本应用当前风格）
        processed_html = process_attack_formulation(html_list, attack_style)

        # 保存结果
        output_file = os.path.join(output_dir, 'adversarial_test',
                                   f'web_test_adv_{letter_mapping[i]}.pkl')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump({
                'html': processed_html,  # 保存完整HTML
                'url': urls,             # 保留原始URL
                'labels': labels         # 标签保持不变
            }, f)
        print(f"已保存对抗样本 {letter_mapping[i]} 至 {output_file}")


def process_single_attack(html, style, temperature=0.8, max_tokens=2048):
    """生成单个对抗性HTML样本"""
    return process_single_html(html, style, temperature, max_tokens)


def process_attack_formulation(html_list, style, temperature=0.8, max_tokens=2048):
    """批量生成对抗性HTML样本"""
    processed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = {
            executor.submit(process_single_attack, html, style, temperature, max_tokens): html
            for html in html_list
        }

        for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(html_list),
                desc="生成对抗性HTML样本"
        ):
            try:
                processed_html = future.result()
                processed.append(processed_html)
            except Exception as e:
                print(f"处理失败: {str(e)}")

    return processed


import numpy as np
import re


def get_phishing_features(html):
    """从HTML中提取钓鱼网站特征标签"""
    # 假设clean_html函数已定义
    cleaned_html = clean_html(html)

    if not cleaned_html:
        print("清洗后的HTML为空，无法提取特征")
        return np.array([0, 0, 0, 0])

    messages = [
        {"role": "user",
         "content": "Analyze the following HTML content (including text and structure) "
                    "and return a list of 4 elements. Each element indicates the presence (1) "
                    "or absence (0) of a phishing characteristic:\n"
                    "1. Urgent requests for personal information or actions\n"
                    "2. Poor grammar, spelling errors, or unprofessional language\n"
                    "3. Suspicious links, fake URLs, or misleading domain names\n"
                    "4. Fake logos, brand impersonation, or false authority claims\n"
                    "Return only the list of 0s and 1s, e.g., [1,0,1,0]. Do not return any other text.\n"
                    f"HTML content: {cleaned_html}"},
        {"role": "assistant", "content": "I will return only the list as requested."}
    ]

    custom_params = {
        'model': MODEL_NAME,
        'messages': messages,
        'thinking': {"type": "disabled"},
        'stream': False,
        'max_tokens': 100,
        'temperature': 0
    }

    try:
        response = client.chat.completions.create(**custom_params)
        result = response.choices[0].message.content.strip()

        # 新的正则表达式匹配逻辑
        match = re.search(r'\[(\d,\s*\d,\s*\d,\s*\d)\]', result)
        if match:
            numbers = match.group(1).split(',')
            return np.array([float(num.strip()) for num in numbers])
        else:
            print(f"特征提取结果格式错误: {result}")
            return np.array([0, 0, 0, 0])

    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return np.array([0, 0, 0, 0])


def generate_phishing_features(output_dir, style_pairs):
    """生成钓鱼网站细粒度特征标签"""

    def process_texts_concurrently(texts, desc, max_workers=50):
        features = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(get_phishing_features, text) for text in texts]
            for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(texts),
                    desc=desc
            ):
                try:
                    features.append(future.result())
                except Exception as e:
                    print(f"处理文本时出错: {str(e)}")
                    features.append(np.array([0, 0, 0, 0]))
        return features

    for mainstream, deceptive in style_pairs:
        mainstream_path = os.path.join(output_dir, 'reframings', f'web_train_{mainstream}.pkl')
        deceptive_path = os.path.join(output_dir, 'reframings', f'web_train_{deceptive}.pkl')

        if not os.path.exists(mainstream_path) or not os.path.exists(deceptive_path):
            print(f"风格数据文件缺失: {mainstream_path} 或 {deceptive_path}")
            continue

        with open(mainstream_path, 'rb') as f:
            mainstream_data = pickle.load(f)
        with open(deceptive_path, 'rb') as f:
            deceptive_data = pickle.load(f)

        # 加载原始HTML
        orig_path = os.path.join(output_dir, 'web_articles', 'web_train.pkl')
        with open(orig_path, 'rb') as f:
            orig_data = pickle.load(f)
        orig_htmls = orig_data.get('html', [])

        # 并发提取细粒度特征
        orig_fg = process_texts_concurrently(orig_htmls, "提取原始HTML特征")
        mainstream_fg = process_texts_concurrently(
            mainstream_data['rewritten'],
            f"提取{mainstream}风格HTML特征"
        )
        deceptive_fg = process_texts_concurrently(
            deceptive_data['rewritten'],
            f"提取{deceptive}风格HTML特征"
        )

        # 保存特征数据
        output_subdir = os.path.join(output_dir, 'veracity_attributions')
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, f'web_fake_standards_{mainstream}_{deceptive}.pkl')

        with open(output_path, 'wb') as f:
            pickle.dump({
                'orig_fg': np.array(orig_fg),
                'standard_fg': np.array(mainstream_fg),
                'deceptive_fg': np.array(deceptive_fg),
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
    read_and_process_htmls('', output_directory)

    # 2. 生成对抗性测试数据
    process_attack_formulation_read_and_process_pkl('', output_directory)

    # 3. 生成细粒度特征标签
    generate_phishing_features(
        output_directory,
        style_pairs=[('neutral', 'sensational'), ('objective', 'emotionally_triggering')]
    )
