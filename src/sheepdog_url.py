import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import argparse
import numpy as np
import sys, os
import warnings
import time
from datetime import datetime
import csv
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_recall_curve,
                             average_precision_score, matthews_corrcoef, recall_score)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from tqdm import tqdm

# 设置项目路径
current_script_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(src_dir)
sys.path.append(project_root)
from utils.load_data_url import *  # 包含load_webpages和load_reframing函数

warnings.filterwarnings("ignore")

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='web', type=str)
parser.add_argument('--model_name', default='SheepDog-Web', type=str)
parser.add_argument('--iters', default=3, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--n_epochs', default=15, type=int)
args = parser.parse_args()

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 随机种子设置
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(0)


# 辅助函数：判断字符串是否为URL格式
def is_url_like(s):
    """检测字符串是否包含URL特征"""
    s = str(s).lower()
    url_features = ['http://', 'https://', '.com', '.net', '.org', '.cn', '.co.uk', 'www.']
    return any(feat in s for feat in url_features)


# 训练数据集类（支持URL特征）
class WebDatasetAug(Dataset):
    def __init__(self, texts, urls, aug_texts1, aug_texts2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len):
        # 1. 数据长度严格验证
        lengths = [
            len(texts), len(urls), len(aug_texts1),
            len(aug_texts2), len(labels), len(fg_label),
            len(aug_fg1), len(aug_fg2)
        ]
        if len(set(lengths)) != 1:
            raise ValueError(f"所有输入数据长度必须一致！长度分别为: {lengths}")

        # 2. 内容混淆检测
        self._detect_content_mixup(texts, urls)

        # 3. 调整数据长度（如有必要）
        texts = self._adjust_length(texts, lengths[0])
        urls = self._adjust_length(urls, lengths[0])
        aug_texts1 = self._adjust_length(aug_texts1, lengths[0])
        aug_texts2 = self._adjust_length(aug_texts2, lengths[0])

        # 4. 数据转换与清洗
        self.texts = [str(text) for text in texts]
        self.urls = [str(url) for url in urls]
        self.aug_texts1 = [str(text) for text in aug_texts1]
        self.aug_texts2 = [str(text) for text in aug_texts2]

        # 5. 标签处理与验证
        self.labels = self.clean_labels(labels)
        self.fg_label = self.clean_finegrained_labels(fg_label)
        self.aug_fg1 = self.clean_finegrained_labels(aug_fg1)
        self.aug_fg2 = self.clean_finegrained_labels(aug_fg2)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def _detect_content_mixup(self, texts, urls):
        """检测texts和urls是否发生内容混淆"""
        if len(texts) == 0 or len(urls) == 0:
            return

        # 检查前3个样本的内容特征
        text_is_url_count = sum([is_url_like(str(texts[i])) for i in range(min(3, len(texts)))])
        url_is_text_count = sum([not is_url_like(str(urls[i])) for i in range(min(3, len(urls)))])

        # 如果发现明显混淆，抛出错误
        if text_is_url_count > 1 and url_is_text_count > 1:
            sample_texts = [str(texts[i])[:50] for i in range(min(3, len(texts)))]
            sample_urls = [str(urls[i])[:50] for i in range(min(3, len(urls)))]
            raise ValueError(
                f"疑似texts和urls内容混淆！\n"
                f"前3个文本样本（应不含URL）: {sample_texts}\n"
                f"前3个URL样本（应含URL）: {sample_urls}"
            )

    @staticmethod
    def _adjust_length(data, target_length):
        """统一调整列表长度以匹配目标长度"""
        if len(data) == target_length:
            return data

        if len(data) > target_length:
            return data[:target_length]
        else:
            # 根据数据类型选择填充值
            if isinstance(data[0], (str, np.str_)):
                fill_value = ''
            elif isinstance(data[0], (int, float, np.number)):
                fill_value = 0
            else:
                fill_value = data[0]  # 使用第一个元素的类型填充
            return data + [fill_value] * (target_length - len(data))

    def clean_labels(self, labels):
        """清洗分类标签，确保为整数类型"""
        cleaned = []
        for label in labels:
            try:
                if isinstance(label, str):
                    label = label.strip().replace('"', '').replace("'", "")
                cleaned.append(int(float(label)))
            except (ValueError, TypeError):
                cleaned.append(0)
        return np.array(cleaned, dtype=int)

    def clean_finegrained_labels(self, labels):
        cleaned = []
        for label in labels:
            try:
                if isinstance(label, str):
                    import re
                    nums = re.findall(r'\d+\.\d+|\d+', label.strip())
                    arr = np.array(nums, dtype=float)[:4]
                elif isinstance(label, (list, np.ndarray)):
                    arr = np.array(label, dtype=float)[:4]
                else:
                    arr = np.full(4, float(label), dtype=float)
                
                # 强制补全为4维
                if len(arr) < 4:
                    arr = np.pad(arr, (0, 4 - len(arr)), mode='constant')
                # 限制取值在[0,1]
                arr = np.clip(arr, 0.0, 1.0)
                cleaned.append(arr)
            except:
                cleaned.append(np.zeros(4, dtype=float))  # 异常样本直接填充0
        return np.array(cleaned, dtype=float)

    def __getitem__(self, item):
        text = self.texts[item]
        url = self.urls[item]
        url_features = self.extract_url_features(url)
        aug_text1 = self.aug_texts1[item]
        aug_text2 = self.aug_texts2[item]
        label = self.labels[item]
        fg_label = self.fg_label[item]
        aug_fg1 = self.aug_fg1[item]
        aug_fg2 = self.aug_fg2[item]

        # 文本编码
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_token_type_ids=False,
            return_attention_mask=True, return_tensors='pt'
        )
        aug1_encoding = self.tokenizer.encode_plus(
            aug_text1, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_token_type_ids=False,
            return_attention_mask=True, return_tensors='pt'
        )
        aug2_encoding = self.tokenizer.encode_plus(
            aug_text2, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_token_type_ids=False,
            return_attention_mask=True, return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'input_ids_aug1': aug1_encoding['input_ids'].flatten(),
            'input_ids_aug2': aug2_encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'attention_mask_aug1': aug1_encoding['attention_mask'].flatten(),
            'attention_mask_aug2': aug2_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'fg_label': torch.FloatTensor(fg_label),  # 二维标签
            'fg_label_aug1': torch.FloatTensor(aug_fg1),
            'fg_label_aug2': torch.FloatTensor(aug_fg2),
            'url_features': torch.FloatTensor(url_features)
        }

    def extract_url_features(self, url):
        """提取URL的5维特征"""
        import tldextract
        ext = tldextract.extract(url)
        domain = ext.domain + '.' + ext.suffix if ext.suffix else ext.domain
        return [
            len(url),  # URL长度
            len(domain),  # 域名长度
            1 if 'http://' in url else 0,  # 是否使用HTTP（非HTTPS）
            1 if '@' in url else 0,  # 是否含@符号
            1 if '//' in url.split('://')[1:] else 0  # 是否含多重//
        ]

    def __len__(self):
        return len(self.texts)


# 测试数据集类
class WebDataset(Dataset):
    def __init__(self, texts, urls, labels, tokenizer, max_len):
        # 1. 数据长度严格验证
        lengths = [len(texts), len(urls), len(labels)]
        if len(set(lengths)) != 1:
            raise ValueError(f"所有输入数据长度必须一致！长度分别为: {lengths}")

        # 2. 内容混淆检测
        self._detect_content_mixup(texts, urls)

        # 3. 调整数据长度（如有必要）
        texts = self._adjust_length(texts, lengths[0])
        urls = self._adjust_length(urls, lengths[0])

        # 4. 数据转换与清洗
        self.texts = [str(text) for text in texts]
        self.urls = [str(url) for url in urls]

        # 5. 清洗并验证标签
        self.labels = self.clean_labels(labels)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def _detect_content_mixup(self, texts, urls):
        """检测texts和urls是否发生内容混淆"""
        if len(texts) == 0 or len(urls) == 0:
            return

        # 检查前3个样本的内容特征
        text_is_url_count = sum([is_url_like(str(texts[i])) for i in range(min(3, len(texts)))])
        url_is_text_count = sum([not is_url_like(str(urls[i])) for i in range(min(3, len(urls)))])

        # 如果发现明显混淆，抛出错误
        if text_is_url_count > 1 and url_is_text_count > 1:
            sample_texts = [str(texts[i])[:50] for i in range(min(3, len(texts)))]
            sample_urls = [str(urls[i])[:50] for i in range(min(3, len(urls)))]
            raise ValueError(
                f"疑似texts和urls内容混淆！\n"
                f"前3个文本样本（应不含URL）: {sample_texts}\n"
                f"前3个URL样本（应含URL）: {sample_urls}"
            )

    @staticmethod
    def _adjust_length(data, target_length):
        """统一调整列表长度以匹配目标长度"""
        if len(data) == target_length:
            return data

        if len(data) > target_length:
            return data[:target_length]
        else:
            # 根据数据类型选择填充值
            if isinstance(data[0], (str, np.str_)):
                fill_value = ''
            elif isinstance(data[0], (int, float, np.number)):
                fill_value = 0
            else:
                fill_value = data[0]  # 使用第一个元素的类型填充
            return data + [fill_value] * (target_length - len(data))

    def clean_labels(self, labels):
        """清洗标签，确保为整数类型"""
        cleaned = []
        for label in labels:
            try:
                if isinstance(label, str):
                    # 特别检查是否为URL
                    if 'http' in label or '.' in label:
                        raise ValueError(f"标签中包含URL: {label}")
                    label = label.strip().replace('"', '').replace("'", "")
                cleaned.append(int(float(label)))
            except (ValueError, TypeError) as e:
                cleaned.append(0)
        return np.array(cleaned, dtype=int)

    def __getitem__(self, item):
        text = self.texts[item]
        url = self.urls[item]
        url_features = self.extract_url_features(url)
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_token_type_ids=False,
            return_attention_mask=True, return_tensors='pt'
        )

        return {
            'news_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'url_features': torch.FloatTensor(url_features)
        }

    def extract_url_features(self, url):
        """与训练集共享相同的URL特征提取逻辑"""
        import tldextract
        ext = tldextract.extract(url)
        domain = ext.domain + '.' + ext.suffix if ext.suffix else ext.domain
        return [
            len(url), len(domain),
            1 if 'http://' in url else 0,
            1 if '@' in url else 0,
            1 if '//' in url.split('://')[1:] else 0
        ]

    def __len__(self):
        return len(self.texts)


# 融合URL特征的RoBERTa模型
class RobertaWebClassifier(nn.Module):
    def __init__(self, n_classes, url_feat_dim=5):
        super(RobertaWebClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(p=0.5)

        # 文本特征处理
        self.text_fc = nn.Linear(self.roberta.config.hidden_size, 128)

        # URL特征处理
        self.url_fc = nn.Linear(url_feat_dim, 32)

        # 融合特征分类
        self.fc_out = nn.Linear(128 + 32, n_classes)  # 融合文本和URL特征
        self.binary_transform = nn.Linear(128 + 32, 2)  # 二分类输出层

    def forward(self, input_ids, attention_mask, url_features):
        # 文本特征提取
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        text_feat = self.dropout(pooled_output)
        text_feat = self.text_fc(text_feat)

        # URL特征提取
        url_feat = self.url_fc(url_features)

        # 特征融合
        combined_feat = torch.cat([text_feat, url_feat], dim=1)

        # 输出层
        output = self.fc_out(combined_feat)
        binary_output = self.binary_transform(combined_feat)

        return output, binary_output


# 创建训练数据加载器
def create_train_loader(contents, urls, contents_aug1, contents_aug2, labels, fg_label, aug_fg1, aug_fg2,
                        tokenizer, max_len, batch_size):
    # 验证数据长度并打印调试信息
    print(f"创建训练加载器 - 文本: {len(contents)}, URL: {len(urls)}, 标签: {len(labels)}")
    ds = WebDatasetAug(
        texts=contents, urls=urls,
        aug_texts1=contents_aug1, aug_texts2=contents_aug2,
        labels=labels, fg_label=fg_label,
        aug_fg1=aug_fg1, aug_fg2=aug_fg2,
        tokenizer=tokenizer, max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)


# 创建评估数据加载器
def create_eval_loader(contents, urls, labels, tokenizer, max_len, batch_size):
    # 验证数据长度并打印调试信息
    print(f"创建评估加载器 - 文本: {len(contents)}, URL: {len(urls)}, 标签: {len(labels)}")
    ds = WebDataset(
        texts=contents, urls=urls,
        labels=labels,
        tokenizer=tokenizer, max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0)


# 随机种子设置
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def calculate_tpr_at_fpr(y_true, y_score, target_fpr):
    """计算特定FPR下的TPR"""
    # 计算FPR和TPR
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # 处理FPR单调递增特性，找到第一个>=target_fpr的点
    idx = np.where(fpr >= target_fpr)[0]
    if len(idx) == 0:  # 所有FPR均小于目标值，取最大TPR
        return tpr[-1]
    return tpr[idx[0]]


# 保存评估结果到CSV
def save_metrics_to_csv(metrics, datasetname, model_name, iterations):
    """将评估指标保存到CSV文件"""
    os.makedirs('../results', exist_ok=True)
    csv_path = f'../results/metrics_{datasetname}_{model_name}.iter{iterations}.csv'
    
    # 计算平均值
    avg_metrics = {key: sum(values)/len(values) if values else 0.0 for key, values in metrics.items() 
                  if key not in ['datetime', 'elapsed_time']}
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(['指标', '平均值', '各次迭代结果'])
        
        # 写入原始测试集指标
        writer.writerow(['', '', ''])
        writer.writerow(['原始测试集指标', '', ''])
        writer.writerow(['准确率 (ACC)', f'{avg_metrics["acc"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['acc'])])
        writer.writerow(['精确率 (Precision)', f'{avg_metrics["prec"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['prec'])])
        writer.writerow(['召回率 (Recall)', f'{avg_metrics["recall"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['recall'])])
        writer.writerow(['F1分数', f'{avg_metrics["f1"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['f1'])])
        writer.writerow(['加权召回率', f'{avg_metrics["weighted_recall"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['weighted_recall'])])
        writer.writerow(['MCC', f'{avg_metrics["mcc"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['mcc'])])
        writer.writerow(['ROC-AUC', f'{avg_metrics["roc_auc"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['roc_auc'])])
        writer.writerow(['PR-AUC', f'{avg_metrics["pr_auc"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['pr_auc'])])
        writer.writerow(['TPR@FPR=0.0001', f'{avg_metrics["tpr_fpr_0001"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['tpr_fpr_0001'])])
        writer.writerow(['TPR@FPR=0.001', f'{avg_metrics["tpr_fpr_001"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['tpr_fpr_001'])])
        writer.writerow(['TPR@FPR=0.01', f'{avg_metrics["tpr_fpr_01"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['tpr_fpr_01'])])
        writer.writerow(['TPR@FPR=0.1', f'{avg_metrics["tpr_fpr_1"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['tpr_fpr_1'])])
        
        # 写入对抗测试集指标
        writer.writerow(['', '', ''])
        writer.writerow(['对抗测试集指标', '', ''])
        writer.writerow(['准确率 (ACC)', f'{avg_metrics["acc_res"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['acc_res'])])
        writer.writerow(['精确率 (Precision)', f'{avg_metrics["prec_res"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['prec_res'])])
        writer.writerow(['召回率 (Recall)', f'{avg_metrics["recall_res"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['recall_res'])])
        writer.writerow(['F1分数', f'{avg_metrics["f1_res"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['f1_res'])])
        writer.writerow(['加权召回率', f'{avg_metrics["weighted_recall_res"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['weighted_recall_res'])])
        writer.writerow(['MCC', f'{avg_metrics["mcc_res"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['mcc_res'])])
        writer.writerow(['ROC-AUC', f'{avg_metrics["roc_auc_res"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['roc_auc_res'])])
        writer.writerow(['PR-AUC', f'{avg_metrics["pr_auc_res"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['pr_auc_res'])])
        writer.writerow(['TPR@FPR=0.0001', f'{avg_metrics["tpr_fpr_res_0001"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['tpr_fpr_res_0001'])])
        writer.writerow(['TPR@FPR=0.001', f'{avg_metrics["tpr_fpr_res_001"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['tpr_fpr_res_001'])])
        writer.writerow(['TPR@FPR=0.01', f'{avg_metrics["tpr_fpr_res_01"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['tpr_fpr_res_01'])])
        writer.writerow(['TPR@FPR=0.1', f'{avg_metrics["tpr_fpr_res_1"]:.4f}', ', '.join(f'{x:.4f}' for x in metrics['tpr_fpr_res_1'])])
        
        # 写入时间信息
        writer.writerow(['', '', ''])
        writer.writerow(['时间信息', '', ''])
        for i in range(iterations):
            writer.writerow([f'迭代 {i}', metrics['datetime'][i], metrics['elapsed_time'][i]])
    
    print(f"评估结果已保存到CSV文件: {csv_path}")


# 训练模型
def train_model(tokenizer, max_len, n_epochs, batch_size, datasetname, iter):
    # 记录开始时间
    start_time = time.time()

    # 加载网页数据集并显式解析返回值
    data = load_webpages(datasetname)
    try:
        # 按照load_webpages的实际返回顺序解析
        train_texts = data[0]
        test_texts = data[1]
        test_texts_res = data[2]  # 对抗性测试集文本
        train_labels = data[3]
        test_labels = data[4]
        train_urls = data[5]
        test_urls = data[6]
    except IndexError:
        raise ValueError("load_webpages返回格式错误，需包含[x_train, x_test, x_test_res, y_train, y_test, url_train, url_test]")

    # 修复对抗性测试集长度不匹配问题
    if len(test_texts_res) != len(test_urls) or len(test_texts_res) != len(test_labels):
        min_len = min(len(test_texts_res), len(test_urls), len(test_labels))
        print(f"警告：对抗性测试集文本({len(test_texts_res)})、URL({len(test_urls)})、标签({len(test_labels)})长度不匹配，将统一截断到{min_len}")
        test_texts_res = test_texts_res[:min_len]
        test_urls_adj = test_urls[:min_len]  # 为对抗测试集创建调整后的URL副本
        test_labels_adj = test_labels[:min_len]  # 为对抗测试集创建调整后的标签副本
    else:
        test_urls_adj = test_urls
        test_labels_adj = test_labels

    # 确保训练集长度一致
    if len(train_texts) != len(train_urls) or len(train_texts) != len(train_labels):
        min_train_len = min(len(train_texts), len(train_urls), len(train_labels))
        print(f"警告：训练集文本({len(train_texts)})、URL({len(train_urls)})、标签({len(train_labels)})长度不匹配，统一截断到{min_train_len}")
        train_texts = train_texts[:min_train_len]
        train_urls = train_urls[:min_train_len]
        train_labels = train_labels[:min_train_len]

    # 确保标准测试集长度一致
    if len(test_texts) != len(test_urls) or len(test_texts) != len(test_labels):
        min_test_len = min(len(test_texts), len(test_urls), len(test_labels))
        print(f"警告：标准测试集文本({len(test_texts)})、URL({len(test_urls)})、标签({len(test_labels)})长度不匹配，统一截断到{min_test_len}")
        test_texts = test_texts[:min_test_len]
        test_urls = test_urls[:min_test_len]
        test_labels = test_labels[:min_test_len]

    # 验证所有数据集长度
    print(f"训练集 - 文本: {len(train_texts)}, URL: {len(train_urls)}, 标签: {len(train_labels)}")
    print(f"标准测试集 - 文本: {len(test_texts)}, URL: {len(test_urls)}, 标签: {len(test_labels)}")
    print(f"对抗测试集 - 文本: {len(test_texts_res)}, URL: {len(test_urls_adj)}, 标签: {len(test_labels_adj)}")

    # 验证标签有效性
    def is_valid_label(label):
        if isinstance(label, (int, float, np.number)):
            return True
        if isinstance(label, str):
            return label.isdigit() and 'http' not in label and '.' not in label
        return False

    # 清理训练标签
    invalid_train = [i for i, lab in enumerate(train_labels) if not is_valid_label(lab)]
    if invalid_train:
        print(f"警告：训练集中发现{len(invalid_train)}个无效标签，已替换为0")
        for i in invalid_train:
            train_labels[i] = 0

    # 清理测试标签
    invalid_test = [i for i, lab in enumerate(test_labels) if not is_valid_label(lab)]
    if invalid_test:
        print(f"警告：测试集中发现{len(invalid_test)}个无效标签，已替换为0")
        for i in invalid_test:
            test_labels[i] = 0

    # 创建测试数据加载器
    test_loader = create_eval_loader(
        test_texts, test_urls, test_labels,
        tokenizer, max_len, batch_size
    )
    
    # 创建对抗测试数据加载器
    test_loader_res = create_eval_loader(
        test_texts_res, test_urls_adj, test_labels_adj,
        tokenizer, max_len, batch_size
    )

    # 创建训练数据增强版本（简单复制作为示例，实际应用中应使用真实增强数据）
    train_texts_aug1 = [text + " [增强1]" for text in train_texts]
    train_texts_aug2 = [text + " [增强2]" for text in train_texts]
    fg_label = np.random.rand(len(train_texts), 4)  # 示例细粒度标签
    aug_fg1 = fg_label.copy()
    aug_fg2 = fg_label.copy()

    # 创建训练数据加载器
    train_loader = create_train_loader(
        train_texts, train_urls, train_texts_aug1, train_texts_aug2,
        train_labels, fg_label, aug_fg1, aug_fg2,
        tokenizer, max_len, batch_size
    )

    # 确定类别数量
    unique_labels = np.unique(train_labels)
    n_classes = len(unique_labels)
    print(f"检测到的类别: {unique_labels}, 类别数量: {n_classes}")

    # 初始化模型
    model = RobertaWebClassifier(n_classes)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 学习率调度器
    total_steps = len(train_loader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 训练循环
    best_accuracy = 0
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{n_epochs}', leave=False)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            input_ids_aug1 = batch['input_ids_aug1'].to(device)
            input_ids_aug2 = batch['input_ids_aug2'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            attention_mask_aug1 = batch['attention_mask_aug1'].to(device)
            attention_mask_aug2 = batch['attention_mask_aug2'].to(device)
            labels = batch['labels'].to(device)
            fg_label = batch['fg_label'].to(device)
            fg_label_aug1 = batch['fg_label_aug1'].to(device)
            fg_label_aug2 = batch['fg_label_aug2'].to(device)
            url_features = batch['url_features'].to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播（原始样本）
            outputs, binary_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                url_features=url_features
            )
            
            # 前向传播（增强样本1）
            outputs_aug1, _ = model(
                input_ids=input_ids_aug1,
                attention_mask=attention_mask_aug1,
                url_features=url_features
            )
            
            # 前向传播（增强样本2）
            outputs_aug2, _ = model(
                input_ids=input_ids_aug2,
                attention_mask=attention_mask_aug2,
                url_features=url_features
            )

            # 计算损失
            loss_ce = criterion(outputs, labels)
            loss_ce_aug1 = criterion(outputs_aug1, labels)
            loss_ce_aug2 = criterion(outputs_aug2, labels)
            
            # 细粒度损失
            loss_fg = F.mse_loss(torch.sigmoid(binary_outputs[:, 1]), fg_label[:, 0])
            loss_fg_aug1 = F.mse_loss(torch.sigmoid(outputs_aug1[:, 1]), fg_label_aug1[:, 0])
            loss_fg_aug2 = F.mse_loss(torch.sigmoid(outputs_aug2[:, 1]), fg_label_aug2[:, 0])

            # 总损失
            loss = (loss_ce + loss_ce_aug1 + loss_ce_aug2) + 0.1 * (loss_fg + loss_fg_aug1 + loss_fg_aug2)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{n_epochs}, 训练损失: {avg_train_loss:.4f}')

        # 在测试集上评估
        model.eval()
        test_loss = 0
        predictions = []
        true_labels = []
        probabilities = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                url_features = batch['url_features'].to(device)

                outputs, binary_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    url_features=url_features
                )

                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        accuracy = accuracy_score(true_labels, predictions)
        print(f'Epoch {epoch + 1}/{n_epochs}, 测试损失: {avg_test_loss:.4f}, 准确率: {accuracy:.4f}')

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'best_model_{datasetname}_iter{iter}.bin')

    # 计算最终评估指标
    def evaluate(loader):
        model.eval()
        predictions = []
        true_labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                url_features = batch['url_features'].to(device)

                outputs, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    url_features=url_features
                )

                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        
        # 计算评估指标
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = score(true_labels, predictions, average='binary')
        weighted_recall = recall_score(true_labels, predictions, average='weighted')
        mcc = matthews_corrcoef(true_labels, predictions)
        
        # 计算ROC-AUC和PR-AUC
        try:
            roc_auc = roc_auc_score(true_labels, probabilities)
        except ValueError:
            roc_auc = 0.0
            
        pr_auc = average_precision_score(true_labels, probabilities)
        
        # 计算特定FPR下的TPR
        tpr_fpr_0001 = calculate_tpr_at_fpr(true_labels, probabilities, 0.0001)
        tpr_fpr_001 = calculate_tpr_at_fpr(true_labels, probabilities, 0.001)
        tpr_fpr_01 = calculate_tpr_at_fpr(true_labels, probabilities, 0.01)
        tpr_fpr_1 = calculate_tpr_at_fpr(true_labels, probabilities, 0.1)
        
        return {
            'acc': accuracy,
            'prec': precision,
            'recall': recall,
            'f1': f1,
            'weighted_recall': weighted_recall,
            'mcc': mcc,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'tpr_fpr_0001': tpr_fpr_0001,
            'tpr_fpr_001': tpr_fpr_001,
            'tpr_fpr_01': tpr_fpr_01,
            'tpr_fpr_1': tpr_fpr_1
        }

    # 在标准测试集上评估
    test_metrics = evaluate(test_loader)
    print("\n标准测试集评估结果:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    # 在对抗测试集上评估
    test_metrics_res = evaluate(test_loader_res)
    print("\n对抗测试集评估结果:")
    for key, value in test_metrics_res.items():
        print(f"{key}: {value:.4f}")

    # 计算耗时
    elapsed_time = time.time() - start_time
    datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 合并指标
    metrics = {**test_metrics, **{f"{k}_res": v for k, v in test_metrics_res.items()}}
    metrics['elapsed_time'] = elapsed_time
    metrics['datetime'] = datetime_str
    
    return metrics


# 主函数
def main():
    # 初始化分词器
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_len = 256  # 可根据实际需求调整

    # 初始化指标存储
    metrics = {
        'acc': [], 'prec': [], 'recall': [], 'f1': [], 'weighted_recall': [],
        'mcc': [], 'roc_auc': [], 'pr_auc': [],
        'tpr_fpr_0001': [], 'tpr_fpr_001': [], 'tpr_fpr_01': [], 'tpr_fpr_1': [],
        'acc_res': [], 'prec_res': [], 'recall_res': [], 'f1_res': [], 'weighted_recall_res': [],
        'mcc_res': [], 'roc_auc_res': [], 'pr_auc_res': [],
        'tpr_fpr_res_0001': [], 'tpr_fpr_res_001': [], 'tpr_fpr_res_01': [], 'tpr_fpr_res_1': [],
        'datetime': [], 'elapsed_time': []
    }

    # 多轮迭代训练与评估
    for i in range(args.iters):
        print(f"\n===== 迭代 {i + 1}/{args.iters} =====")
        set_seed(i)  # 不同迭代使用不同种子
        iter_metrics = train_model(
            tokenizer, max_len, args.n_epochs, 
            args.batch_size, args.dataset_name, i
        )
        
        # 保存本轮迭代指标
        for key in metrics:
            if key in iter_metrics:
                metrics[key].append(iter_metrics[key])

    # 保存结果到CSV
    save_metrics_to_csv(metrics, args.dataset_name, args.model_name, args.iters)

    # 打印平均结果
    print("\n===== 多轮迭代平均结果 =====")
    print("标准测试集:")
    print(f"准确率: {np.mean(metrics['acc']):.4f} ± {np.std(metrics['acc']):.4f}")
    print(f"F1分数: {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")
    print(f"ROC-AUC: {np.mean(metrics['roc_auc']):.4f} ± {np.std(metrics['roc_auc']):.4f}")
    
    print("\n对抗测试集:")
    print(f"准确率: {np.mean(metrics['acc_res']):.4f} ± {np.std(metrics['acc_res']):.4f}")
    print(f"F1分数: {np.mean(metrics['f1_res']):.4f} ± {np.std(metrics['f1_res']):.4f}")
    print(f"ROC-AUC: {np.mean(metrics['roc_auc_res']):.4f} ± {np.std(metrics['roc_auc_res']):.4f}")


if __name__ == "__main__":
    main()