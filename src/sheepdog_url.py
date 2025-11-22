import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, auc, matthews_corrcoef
)
import warnings
import csv
import pickle
import re
from urllib.parse import urlparse
from datetime import datetime
import argparse
warnings.filterwarnings('ignore')

# -------------------------- 全局配置与常量 --------------------------
URL_FEATURE_DIM = 8  # URL特征维度（与提取逻辑一致）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Global] Using device: {DEVICE}")

# 路径配置（可通过命令行参数覆盖）
DEFAULT_DATA_DIR = "../data/web_articles"
DEFAULT_ADV_DATA_DIR = "../data/adversarial_test"
DEFAULT_MODEL_SAVE_PATH = "../models/sheepdog_url_chunks.pth"
DEFAULT_METRICS_SAVE_PATH = "../results/evaluation_metrics.csv"
DEFAULT_PRED_SAVE_DIR = "../results"

# -------------------------- 1. 命令行参数解析 --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Sheepdog URL + 长文本分段钓鱼检测模型（支持完整评估指标）")
    
    # 数据集参数
    parser.add_argument("--dataset_name", type=str, default="web", help="数据集名称（如'politifact'，与pkl文件前缀一致）")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="训练/测试集pkl文件目录")
    parser.add_argument("--adv_data_dir", type=str, default=DEFAULT_ADV_DATA_DIR, help="对抗性测试集pkl文件目录")
    
    # 模型与训练参数
    parser.add_argument("--max_len", type=int, default=512, help="文本分段最大长度（与模型max_len一致）")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小（分段后建议≤2）")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减（正则化）")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "max", "weighted"], help="分段聚合策略")
    parser.add_argument("--val_split", type=float, default=0.1, help="从训练集拆分验证集的比例")
    
    # 保存路径参数
    parser.add_argument("--model_save_path", type=str, default=DEFAULT_MODEL_SAVE_PATH, help="最优模型保存路径")
    parser.add_argument("--metrics_save_path", type=str, default=DEFAULT_METRICS_SAVE_PATH, help="评估指标CSV保存路径")
    parser.add_argument("--pred_save_dir", type=str, default=DEFAULT_PRED_SAVE_DIR, help="预测结果保存目录")
    
    # 其他参数
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader多进程数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    
    args = parser.parse_args()
    return args

# -------------------------- 2. 工具函数（数据分段、URL特征提取） --------------------------
def set_seed(seed):
    """设置随机种子，保证实验可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"[Utils] Random seed set to {seed}")

def split_long_text(text, max_len):
    """将长文本分割为多个不超过max_len的片段"""
    chunks = []
    if not text:
        return [""]
    for i in range(0, len(text), max_len):
        chunk = text[i:i+max_len].strip()
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else [""]

def extract_url_features(url):
    """提取URL的8维特征（标准化）"""
    try:
        parsed = urlparse(url)
        features = []
        
        # 1. 协议类型（http=0, https=1, 其他=2）
        scheme = parsed.scheme.lower()
        features.append(1 if scheme == 'https' else 0 if scheme == 'http' else 2)
        
        # 2. 域名长度（标准化到0-1）
        domain = parsed.netloc
        features.append(min(len(domain), 200) / 200)
        
        # 3. 是否包含多子域名（不含www）
        subdomain_count = len([p for p in domain.split('.') if p and p != 'www'])
        features.append(1 if subdomain_count > 1 else 0)
        
        # 4. 路径长度（标准化到0-1）
        path_len = len(parsed.path)
        features.append(min(path_len, 200) / 200)
        
        # 5. 是否有查询参数
        features.append(1 if parsed.query else 0)
        
        # 6. 是否有锚点
        features.append(1 if parsed.fragment else 0)
        
        # 7. 域名数字占比
        digit_ratio = len(re.findall(r'\d', domain)) / len(domain) if domain else 0.0
        features.append(digit_ratio)
        
        # 8. 是否包含特殊字符
        special_chars = r'@&=+%$#!?*()[]{}'
        features.append(1 if any(c in special_chars for c in url) else 0)
        
        return np.array(features, dtype=np.float32)
    except:
        return np.array([2, 0, 0, 0, 0, 0, 0.0, 0], dtype=np.float32)

# -------------------------- 3. 数据加载相关（Dataset + DataLoader） --------------------------
class WebDatasetWithChunks(Dataset):
    """适配分段文本的Dataset类（保存urls便于后续预测结果保存）"""
    def __init__(self, texts_chunks, urls, labels, tokenizer, max_len):
        self.texts_chunks = texts_chunks  # 格式：[[chunk1, chunk2], ...]
        self.urls = urls  # 保存URL，用于预测结果输出
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts_chunks)

    def __getitem__(self, idx):
        chunks = self.texts_chunks[idx]
        url = self.urls[idx]
        label = self.labels[idx]

        # 对每个片段编码
        chunk_encodings = []
        for chunk in chunks:
            encoding = self.tokenizer.encode_plus(
                chunk,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            chunk_encodings.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            })

        return {
            'chunks': chunk_encodings,
            'url_features': torch.tensor(extract_url_features(url), dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long),
            'url': url  # 单个URL，用于后续保存
        }

def collate_chunks(batch):
    """自定义Collate函数：处理片段数量不一致问题"""
    chunks_list = [item['chunks'] for item in batch]
    url_features = torch.stack([item['url_features'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    urls = [item['url'] for item in batch]  # 收集当前批次的URL
    
    return {
        'chunks': chunks_list,
        'url_features': url_features,
        'labels': labels,
        'urls': urls  # 返回URL列表
    }

def load_pkl_data(file_path):
    """加载pkl文件，验证关键字段"""
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"加载pkl文件失败：{file_path} -> {str(e)}")
    
    required_keys = ['html', 'url', 'labels']
    missing_keys = [k for k in required_keys if k not in data_dict]
    if missing_keys:
        raise KeyError(f"pkl文件缺少关键字段：{missing_keys}（文件：{file_path}）")
    
    return data_dict

def create_dataloaders(args, tokenizer):
    """加载数据并创建训练/验证/测试/对抗性测试集DataLoader"""
    print(f"\n[DataLoader] 开始加载数据（数据集：{args.dataset_name}）")
    
    # 1. 定义文件路径
    train_pkl = os.path.join(args.data_dir, f"{args.dataset_name}_train.pkl")
    test_pkl = os.path.join(args.data_dir, f"{args.dataset_name}_test.pkl")
    adv_pkl = os.path.join(args.adv_data_dir, f"{args.dataset_name}_test_adv_A.pkl")
    
    # 2. 加载pkl数据
    train_dict = load_pkl_data(train_pkl)
    test_dict = load_pkl_data(test_pkl)
    adv_dict = load_pkl_data(adv_pkl)
    
    # 3. 文本分段（核心步骤）
    print(f"[DataLoader] 对长文本进行分段（max_len={args.max_len}）")
    x_train = [split_long_text(text, args.max_len) for text in train_dict['html']]
    x_test = [split_long_text(text, args.max_len) for text in test_dict['html']]
    x_adv_test = [split_long_text(text, args.max_len) for text in adv_dict['html']]
    
    # 4. 提取其他字段
    y_train, url_train = train_dict['labels'], train_dict['url']
    y_test, url_test = test_dict['labels'], test_dict['url']
    y_adv_test, url_adv_test = adv_dict['labels'], adv_dict['url']
    
    # 5. 拆分验证集（从训练集取val_split比例）
    val_size = int(len(x_train) * args.val_split)
    x_val, y_val, url_val = x_train[:val_size], y_train[:val_size], url_train[:val_size]
    x_train, y_train, url_train = x_train[val_size:], y_train[val_size:], url_train[val_size:]
    
    # 6. 打印数据统计
    def print_stats(name, texts_chunks, labels):
        total = len(texts_chunks)
        chunks_total = sum(len(c) for c in texts_chunks)
        pos = sum(1 for l in labels if l == 1)
        print(f"  {name:15s}: 样本数={total:4d} | 正类={pos:4d} | 负类={total-pos:4d} | 总片段数={chunks_total:5d} | 平均片段数={chunks_total/total:.2f}")
    
    print("\n[DataLoader] 数据统计：")
    print_stats("训练集", x_train, y_train)
    print_stats("验证集", x_val, y_val)
    print_stats("测试集", x_test, y_test)
    print_stats("对抗性测试集", x_adv_test, y_adv_test)
    
    # 7. 创建DataLoader
    print(f"\n[DataLoader] 创建DataLoader（batch_size={args.batch_size}）")
    train_loader = DataLoader(
        WebDatasetWithChunks(x_train, url_train, y_train, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_chunks,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        WebDatasetWithChunks(x_val, url_val, y_val, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_chunks,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        WebDatasetWithChunks(x_test, url_test, y_test, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_chunks,
        num_workers=args.num_workers,
        pin_memory=True
    )
    adv_test_loader = DataLoader(
        WebDatasetWithChunks(x_adv_test, url_adv_test, y_adv_test, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_chunks,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"[DataLoader] DataLoader创建完成：")
    print(f"  训练集批次：{len(train_loader)} | 验证集批次：{len(val_loader)} | 测试集批次：{len(test_loader)} | 对抗性测试集批次：{len(adv_test_loader)}")
    
    return train_loader, val_loader, test_loader, adv_test_loader

# -------------------------- 4. 模型定义（分段聚合+URL特征融合） --------------------------
class RobertaWebClassifier(nn.Module):
    def __init__(self, aggregation_strategy='mean'):
        super(RobertaWebClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(
            self.roberta.config.hidden_size + URL_FEATURE_DIM,
            2  # 二分类：合法/钓鱼
        )
        self.sigmoid = nn.Sigmoid()
        assert aggregation_strategy in ['mean', 'max', 'weighted'], "无效的聚合策略"
        self.aggregation_strategy = aggregation_strategy

    def forward(self, chunks, url_features):
        batch_size = len(chunks)
        batch_aggregated = []
        
        for i in range(batch_size):
            sample_chunks = chunks[i]
            chunk_features = []
            
            # 提取每个片段的CLS特征
            for chunk in sample_chunks:
                input_ids = chunk['input_ids'].unsqueeze(0).to(DEVICE)
                attention_mask = chunk['attention_mask'].unsqueeze(0).to(DEVICE)
                
                roberta_out = self.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                cls_feat = roberta_out.last_hidden_state[:, 0, :]  # (1, hidden_size)
                chunk_features.append(cls_feat)
            
            # 片段聚合
            chunk_feat_tensor = torch.cat(chunk_features, dim=0)
            if self.aggregation_strategy == 'mean':
                agg_feat = torch.mean(chunk_feat_tensor, dim=0, keepdim=True)
            elif self.aggregation_strategy == 'max':
                agg_feat = torch.max(chunk_feat_tensor, dim=0, keepdim=True)[0]
            elif self.aggregation_strategy == 'weighted':
                chunk_lengths = [torch.sum(c['attention_mask']).item() for c in sample_chunks]
                weights = torch.tensor(chunk_lengths, dtype=torch.float32).to(DEVICE)
                weights = weights / torch.sum(weights)
                agg_feat = torch.matmul(weights.unsqueeze(0), chunk_feat_tensor)
            
            batch_aggregated.append(agg_feat)
        
        # 融合URL特征并分类
        batch_aggregated = torch.cat(batch_aggregated, dim=0).to(DEVICE)
        url_features = url_features.to(DEVICE)
        combined = torch.cat([batch_aggregated, url_features], dim=1)
        x = self.dropout(combined)
        logits = self.classifier(x)
        probs = self.sigmoid(logits)
        
        return logits, probs

# -------------------------- 5. 评估指标计算（完整16个指标） --------------------------
def calculate_tpr_at_fpr(y_true, y_score, target_fprs):
    """计算指定FPR下的TPR（线性插值）"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    tpr_results = {}
    
    for target_fpr in target_fprs:
        idx = np.searchsorted(fpr, target_fpr, side='left')
        if idx == 0:
            tpr_val = tpr[0] if len(tpr) > 0 else 0.0
        elif idx >= len(fpr):
            tpr_val = tpr[-1] if len(tpr) > 0 else 0.0
        else:
            if fpr[idx] == target_fpr:
                tpr_val = tpr[idx]
            else:
                # 线性插值
                tpr_val = tpr[idx-1] + (target_fpr - fpr[idx-1]) * (tpr[idx] - tpr[idx-1]) / (fpr[idx] - fpr[idx-1])
        tpr_results[f'TPR@FPR={target_fpr}'] = tpr_val
    
    return tpr_results

def compute_all_metrics(y_true, y_pred, y_score):
    """计算所有16个评估指标"""
    # 1. 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 2. 基础指标
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 3. AUC指标
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = 0.0
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec_curve, prec_curve)
    
    # 4. 特定FPR的TPR
    target_fprs = [0.0001, 0.001, 0.01, 0.1]
    tpr_at_fpr = calculate_tpr_at_fpr(y_true, y_score, target_fprs)
    
    # 整合并保留4位小数
    metrics = {
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'ACC': round(acc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'Weighted F1': round(weighted_f1, 4),
        'MCC': round(mcc, 4),
        'ROC-AUC': round(roc_auc, 4),
        'PR-AUC': round(pr_auc, 4),
        **{k: round(v, 4) for k, v in tpr_at_fpr.items()}
    }
    
    return metrics

# -------------------------- 6. 训练与评估函数 --------------------------
def train_epoch(model, loader, optimizer, scheduler, criterion, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(loader, desc=f"Train Epoch {epoch+1:2d}", leave=False)
    for batch in progress_bar:
        chunks = batch['chunks']
        url_features = batch['url_features']
        labels = batch['labels'].to(DEVICE)
        
        optimizer.zero_grad()
        logits, probs = model(chunks, url_features)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # 收集结果
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(loader)
    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"[Train] Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Acc: {train_acc:.4f} | Weighted F1: {train_f1:.4f}")
    
    return avg_loss, train_acc, train_f1

def eval_model(model, loader, criterion, split_name, save_preds=False, save_path=None):
    """评估模型并返回所有指标"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []  # 正类概率
    all_urls = []    # URL列表
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Eval {split_name:15s}", leave=False)
        for batch in progress_bar:
            chunks = batch['chunks']
            url_features = batch['url_features']
            labels = batch['labels'].to(DEVICE)
            urls = batch['urls']  # 从batch中获取URL
            
            logits, probs = model(chunks, url_features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # 收集结果
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            scores = probs[:, 1].cpu().numpy()  # 正类（钓鱼）概率
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores)
            all_urls.extend(urls)
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    
    # 计算指标
    metrics = compute_all_metrics(all_labels, all_preds, all_scores)
    metrics['Average Loss'] = round(total_loss / len(loader), 4)
    
    # 打印结果
    print(f"\n[Eval] {split_name} Results:")
    print("-" * 70)
    for key, value in metrics.items():
        print(f"{key:<20}: {value}")
    print("-" * 70)
    print(f"[Eval] {split_name} Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Legitimate(0)', 'Phishing(1)'], zero_division=0))
    
    # 保存预测结果
    if save_preds and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['URL', 'True_Label', 'Predicted_Label', 'Phishing_Probability'])
            for url, true_lab, pred_lab, score in zip(all_urls, all_labels, all_preds, all_scores):
                writer.writerow([url, true_lab, pred_lab, round(score, 4)])
        print(f"[Eval] 预测结果已保存到：{save_path}\n")
    
    return metrics

# -------------------------- 7. 结果保存函数 --------------------------
def save_metrics_to_csv(args, metrics_dict):
    """将所有数据集的指标保存到CSV"""
    os.makedirs(os.path.dirname(args.metrics_save_path), exist_ok=True)
    
    # 表头：基础信息 + 所有指标
    base_cols = ['Timestamp', 'Dataset', 'Aggregation', 'Max_Len', 'Batch_Size', 'Epochs', 'Split']
    metric_cols = list(next(iter(metrics_dict.values())).keys())
    headers = base_cols + metric_cols
    
    # 检查文件是否存在（不存在则写表头）
    file_exists = os.path.exists(args.metrics_save_path)
    
    with open(args.metrics_save_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        
        # 写入每个分割的指标
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for split, metrics in metrics_dict.items():
            row = {
                'Timestamp': timestamp,
                'Dataset': args.dataset_name,
                'Aggregation': args.aggregation,
                'Max_Len': args.max_len,
                'Batch_Size': args.batch_size,
                'Epochs': args.epochs,
                'Split': split,
                **metrics
            }
            writer.writerow(row)
    
    print(f"\n[Save] 所有评估指标已保存到：{args.metrics_save_path}")

# -------------------------- 8. 主训练流程 --------------------------
def main(args):
    # 1. 初始化设置
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    os.makedirs(args.pred_save_dir, exist_ok=True)
    
    # 2. 加载Tokenizer
    print(f"\n[Model] 加载RoBERTa Tokenizer（max_len={args.max_len}）")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # 3. 创建DataLoader
    train_loader, val_loader, test_loader, adv_test_loader = create_dataloaders(args, tokenizer)
    
    # 4. 初始化模型、损失函数、优化器
    print(f"\n[Model] 初始化模型（聚合策略：{args.aggregation}）")
    model = RobertaWebClassifier(aggregation_strategy=args.aggregation)
    model.to(DEVICE)
    print(f"[Model] 模型参数总数：{sum(p.numel() for p in model.parameters()):,}")
    
    # 类别权重（处理类别不平衡）
    train_labels = [label for batch in train_loader for label in batch['labels'].numpy()]
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor([len(train_labels)/count for count in class_counts], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"[Train] 类别权重：{class_weights.cpu().numpy()}")
    
    # 优化器与调度器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )
    
    # 5. 开始训练
    print(f"\n[Train] 开始训练（共{args.epochs}轮）")
    print("=" * 80)
    best_val_f1 = 0.0
    train_history = []
    val_history = []
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion, epoch)
        train_history.append({'loss': train_loss, 'acc': train_acc, 'f1': train_f1})
        
        # 验证
        val_metrics = eval_model(model, val_loader, criterion, split_name="Validation")
        val_history.append(val_metrics)
        current_val_f1 = val_metrics['F1']
        
        # 保存最优模型（基于验证集F1）
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'args': args,
                'train_history': train_history,
                'val_history': val_history
            }, args.model_save_path)
            print(f"[Save] 最优模型已保存（验证集F1：{best_val_f1:.4f}）\n")
        print("=" * 80)
    
    # 6. 加载最优模型进行最终评估
    print(f"\n[Eval] 加载最优模型进行最终评估（验证集最佳F1：{best_val_f1:.4f}）")
    print("=" * 80)
    checkpoint = torch.load(args.model_save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估所有数据集
    final_metrics = {}
    # 训练集
    final_metrics['Train'] = eval_model(model, train_loader, criterion, split_name="Final_Train")
    # 验证集
    final_metrics['Validation'] = eval_model(model, val_loader, criterion, split_name="Final_Validation")
    # 测试集（保存预测结果）
    test_pred_path = os.path.join(args.pred_save_dir, f"{args.dataset_name}_test_preds.csv")
    final_metrics['Test'] = eval_model(model, test_loader, criterion, split_name="Final_Test", save_preds=True, save_path=test_pred_path)
    # 对抗性测试集（保存预测结果）
    adv_pred_path = os.path.join(args.pred_save_dir, f"{args.dataset_name}_adv_test_preds.csv")
    final_metrics['Adversarial_Test'] = eval_model(model, adv_test_loader, criterion, split_name="Final_Adversarial_Test", save_preds=True, save_path=adv_pred_path)
    
    # 7. 保存所有指标到CSV
    save_metrics_to_csv(args, final_metrics)
    
    # 8. 打印训练总结
    print("\n" + "=" * 80)
    print("训练总结")
    print("=" * 80)
    print(f"数据集：{args.dataset_name}")
    print(f"模型配置：max_len={args.max_len} | batch_size={args.batch_size} | 聚合策略={args.aggregation} | 学习率={args.lr}")
    print(f"最优验证集F1：{best_val_f1:.4f}")
    print(f"测试集关键指标：F1={final_metrics['Test']['F1']:.4f} | ROC-AUC={final_metrics['Test']['ROC-AUC']:.4f} | MCC={final_metrics['Test']['MCC']:.4f}")
    print(f"对抗性测试集F1：{final_metrics['Adversarial_Test']['F1']:.4f}")
    print("=" * 80)

# -------------------------- 9. 入口函数（直接运行） --------------------------
if __name__ == "__main__":
    args = parse_args()
    print("=" * 80)
    print("Sheepdog 钓鱼检测模型（长文本分段+完整评估指标）")
    print("=" * 80)
    print("运行参数：")
    for k, v in vars(args).items():
        print(f"  {k:<20}: {v}")
    print("=" * 80)
    
    try:
        main(args)
        print("\n训练完成！所有结果已保存到指定路径～")
    except Exception as e:
        print(f"\n运行出错：{str(e)}")
        raise e