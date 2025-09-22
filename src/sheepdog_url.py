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
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from tqdm import tqdm

#sys.path.append(os.getcwd())
# 1. 获取当前脚本（sheepdog_url.py）的绝对路径
current_script_path = os.path.abspath(__file__)
# 2. 获取当前脚本所在目录（src文件夹）的路径
src_dir = os.path.dirname(current_script_path)
# 3. 获取src的上级目录（即项目根目录，utils就在这里）
project_root = os.path.dirname(src_dir)
# 4. 将项目根目录加入Python的路径搜索列表
sys.path.append(project_root)
from utils.load_data_url import *  # 包含load_webpages和load_reframing函数

warnings.filterwarnings("ignore")

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='web', type=str)
parser.add_argument('--model_name', default='SheepDog-Web', type=str)
parser.add_argument('--iters', default=3, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--n_epochs', default=5, type=int)
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
                # print(f"警告：无效分类标签 '{label}' 已替换为0")
                cleaned.append(0)
        return np.array(cleaned, dtype=int)

    def clean_finegrained_labels(self, labels):
        """清洗细粒度标签，确保为浮点数类型"""
        cleaned = []
        for label in labels:
            try:
                if isinstance(label, str):
                    label = label.strip().replace('"', '').replace("'", "")
                cleaned.append(float(label))
            except (ValueError, TypeError):
                # print(f"警告：无效细粒度标签 '{label}' 已替换为0.0")
                cleaned.append(0.0)
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
            'fg_label': torch.FloatTensor([fg_label, 1 - fg_label]),  # 二维标签
            'fg_label_aug1': torch.FloatTensor([aug_fg1, 1 - aug_fg1]),
            'fg_label_aug2': torch.FloatTensor([aug_fg2, 1 - aug_fg2]),
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
                # print(f"警告：无效标签 '{label}' 已替换为0 ({str(e)})")
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


# 训练模型
def train_model(tokenizer, max_len, n_epochs, batch_size, datasetname, iter):
    # 加载网页数据集并显式解析返回值
    data = load_webpages(datasetname)
    try:
        # 按照load_webpages的实际返回顺序解析：
        # [训练文本, 测试文本, 对抗测试文本, 训练标签, 测试标签, 训练URL, 测试URL]
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
        contents=test_texts,
        urls=test_urls,
        labels=test_labels,
        tokenizer=tokenizer,
        max_len=max_len,
        batch_size=batch_size
    )
    test_loader_res = create_eval_loader(
        contents=test_texts_res,
        urls=test_urls_adj,  # 使用调整后的URL
        labels=test_labels_adj,  # 使用调整后的标签
        tokenizer=tokenizer,
        max_len=max_len,
        batch_size=batch_size
    )

    # 初始化模型（二分类：钓鱼/正常）
    model = RobertaWebClassifier(n_classes=2).to(device)
    train_losses = []
    train_accs = []
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_texts) * n_epochs // batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 训练循环
    for epoch in range(n_epochs):
        model.train()
        # 加载增强数据
        x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t = load_reframing(args.dataset_name)

        # 调整增强数据长度以匹配训练集
        target_len = len(train_texts)
        if len(x_train_res1) != target_len:
            print(f"警告：增强文本1长度({len(x_train_res1)})与训练文本长度({target_len})不匹配，将调整")
            x_train_res1 = WebDatasetAug._adjust_length(None, x_train_res1, target_len)
            x_train_res2 = WebDatasetAug._adjust_length(None, x_train_res2, target_len)
            y_train_fg = WebDatasetAug._adjust_length(None, y_train_fg, target_len)
            y_train_fg_m = WebDatasetAug._adjust_length(None, y_train_fg_m, target_len)
            y_train_fg_t = WebDatasetAug._adjust_length(None, y_train_fg_t, target_len)

        # 创建训练数据加载器
        train_loader = create_train_loader(
            contents=train_texts,
            urls=train_urls,
            contents_aug1=x_train_res1,
            contents_aug2=x_train_res2,
            labels=train_labels,
            fg_label=y_train_fg,
            aug_fg1=y_train_fg_m,
            aug_fg2=y_train_fg_t,
            tokenizer=tokenizer,
            max_len=max_len,
            batch_size=batch_size
        )

        avg_loss = []
        avg_acc = []
        for Batch_data in tqdm(train_loader):
            # 准备输入数据
            input_ids = Batch_data["input_ids"].to(device)
            attention_mask = Batch_data["attention_mask"].to(device)
            input_ids_aug1 = Batch_data["input_ids_aug1"].to(device)
            attention_mask_aug1 = Batch_data["attention_mask_aug1"].to(device)
            input_ids_aug2 = Batch_data["input_ids_aug2"].to(device)
            attention_mask_aug2 = Batch_data["attention_mask_aug2"].to(device)
            targets = Batch_data["labels"].to(device)

            # 调整细粒度标签的形状以匹配模型输出
            fg_labels = Batch_data["fg_label"].squeeze(1).to(device)  # 从[batch,1,2]变为[batch,2]
            fg_labels_aug1 = Batch_data["fg_label_aug1"].squeeze(1).to(device)
            fg_labels_aug2 = Batch_data["fg_label_aug2"].squeeze(1).to(device)

            # URL特征
            url_features = Batch_data["url_features"].to(device)
            url_features_aug1 = url_features
            url_features_aug2 = url_features

            # 前向传播
            out_labels, out_labels_bi = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                url_features=url_features
            )
            out_labels_aug1, out_labels_bi_aug1 = model(
                input_ids=input_ids_aug1,
                attention_mask=attention_mask_aug1,
                url_features=url_features_aug1
            )
            out_labels_aug2, out_labels_bi_aug2 = model(
                input_ids=input_ids_aug2,
                attention_mask=attention_mask_aug2,
                url_features=url_features_aug2
            )

            # 计算损失
            fg_criterion = nn.BCELoss()
            finegrain_loss = (
                                     fg_criterion(F.sigmoid(out_labels), fg_labels) +
                                     fg_criterion(F.sigmoid(out_labels_aug1), fg_labels_aug1) +
                                     fg_criterion(F.sigmoid(out_labels_aug2), fg_labels_aug2)
                             ) / 3

            sup_criterion = nn.CrossEntropyLoss()
            sup_loss = sup_criterion(out_labels_bi, targets)

            cons_criterion = nn.KLDivLoss(reduction='batchmean')
            out_probs = F.softmax(out_labels_bi, dim=-1)
            cons_loss = 0.5 * (
                    cons_criterion(F.log_softmax(out_labels_bi_aug1, dim=-1), out_probs) +
                    cons_criterion(F.log_softmax(out_labels_bi_aug2, dim=-1), out_probs)
            )

            loss = sup_loss + cons_loss + finegrain_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            scheduler.step()

            # 计算准确率
            _, pred = out_labels_bi.max(dim=-1)
            correct = pred.eq(targets).sum().item()
            train_acc = correct / len(targets)
            avg_acc.append(train_acc)

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        print(f"Iter {iter:03d} | Epoch {epoch:05d} | Train Acc. {np.mean(avg_acc):.4f}")

        # 最后一轮评估
        if epoch == n_epochs - 1:
            model.eval()
            y_pred, y_pred_res, y_test_true, y_test_true_res = [], [], [], []  # 为对抗测试集创建独立的真实标签列表

            # 评估原始测试集
            for Batch_data in tqdm(test_loader):
                with torch.no_grad():
                    input_ids = Batch_data["input_ids"].to(device, non_blocking=True)
                    attention_mask = Batch_data["attention_mask"].to(device, non_blocking=True)
                    url_features = Batch_data["url_features"].to(device, non_blocking=True)
                    targets = Batch_data["labels"].to(device, non_blocking=True)

                    _, val_out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        url_features=url_features
                    )
                    _, val_pred = val_out.max(dim=1)
                    y_pred.append(val_pred)
                    y_test_true.append(targets)  # 原始测试集标签

            # 评估对抗性测试集 - 收集对应标签
            for Batch_data in tqdm(test_loader_res):
                with torch.no_grad():
                    input_ids_aug = Batch_data["input_ids"].to(device, non_blocking=True)
                    attention_mask_aug = Batch_data["attention_mask"].to(device, non_blocking=True)
                    url_features_aug = Batch_data["url_features"].to(device, non_blocking=True)
                    targets_aug = Batch_data["labels"].to(device, non_blocking=True)  # 获取对抗测试集对应的标签

                    _, val_out_aug = model(
                        input_ids=input_ids_aug,
                        attention_mask=attention_mask_aug,
                        url_features=url_features_aug
                    )
                    _, val_pred_aug = val_out_aug.max(dim=1)
                    y_pred_res.append(val_pred_aug)
                    y_test_true_res.append(targets_aug)  # 保存对抗测试集对应的标签

            # 计算指标 - 使用各自对应的标签
            y_pred = torch.cat(y_pred, dim=0)
            y_test_true = torch.cat(y_test_true, dim=0)

            # 对抗测试集使用自己的标签列表
            y_pred_res = torch.cat(y_pred_res, dim=0)
            y_test_true_res = torch.cat(y_test_true_res, dim=0)

            # 验证长度一致性
            if len(y_test_true) != len(y_pred):
                raise ValueError(f"原始测试集标签与预测长度不匹配: {len(y_test_true)} vs {len(y_pred)}")
            if len(y_test_true_res) != len(y_pred_res):
                raise ValueError(f"对抗测试集标签与预测长度不匹配: {len(y_test_true_res)} vs {len(y_pred_res)}")

            # 计算原始测试集指标
            acc = accuracy_score(
                y_test_true.detach().cpu().numpy(),
                y_pred.detach().cpu().numpy()
            )
            precision, recall, fscore, _ = score(
                y_test_true.detach().cpu().numpy(),
                y_pred.detach().cpu().numpy(),
                average='macro'
            )

            # 计算对抗测试集指标（使用对应的标签）
            acc_res = accuracy_score(
                y_test_true_res.detach().cpu().numpy(),  # 使用对抗测试集自己的标签
                y_pred_res.detach().cpu().numpy()
            )
            precision_res, recall_res, fscore_res, _ = score(
                y_test_true_res.detach().cpu().numpy(),  # 使用对抗测试集自己的标签
                y_pred_res.detach().cpu().numpy(),
                average='macro'
            )

    # 保存模型
    os.makedirs('../checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'../checkpoints/{datasetname}_iter{iter}.pth')

    print(f"-----------------End of Iter {iter:03d}-----------------")
    print([f'Global Test Accuracy:{acc:.4f}',
           f'Precision:{precision:.4f}',
           f'Recall:{recall:.4f}',
           f'F1:{fscore:.4f}'])
    print("-----------------Restyle-----------------")
    print([f'Global Test Accuracy:{acc_res:.4f}',
           f'Precision:{precision_res:.4f}',
           f'Recall:{recall_res:.4f}',
           f'F1:{fscore_res:.4f}'])

    return acc, precision, recall, fscore, acc_res, precision_res, recall_res, fscore_res


if __name__ == '__main__':
    # 增加环境变量设置以优化CUDA内存分配
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    datasetname = args.dataset_name
    batch_size = args.batch_size
    max_len = 512
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    n_epochs = args.n_epochs
    iterations = args.iters

    # 指标存储
    test_accs = []
    prec_all, rec_all, f1_all = [], [], []
    test_accs_res = []
    prec_all_res, rec_all_res, f1_all_res = [], [], []

    for iter in range(iterations):
        set_seed(iter)
        acc, prec, recall, f1, acc_res, prec_res, recall_res, f1_res = train_model(
            tokenizer, max_len, n_epochs, batch_size, datasetname, iter
        )

        test_accs.append(acc)
        prec_all.append(prec)
        rec_all.append(recall)
        f1_all.append(f1)
        test_accs_res.append(acc_res)
        prec_all_res.append(prec_res)
        rec_all_res.append(recall_res)
        f1_all_res.append(f1_res)

    # 输出平均结果
    print(
        f"Total_Test_Accuracy: {sum(test_accs) / iterations:.4f}|Prec_Macro: {sum(prec_all) / iterations:.4f}|Rec_Macro: {sum(rec_all) / iterations:.4f}|F1_Macro: {sum(f1_all) / iterations:.4f}")
    print(
        f"Restyle_Test_Accuracy: {sum(test_accs_res) / iterations:.4f}|Prec_Macro: {sum(prec_all_res) / iterations:.4f}|Rec_Macro: {sum(rec_all_res) / iterations:.4f}|F1_Macro: {sum(f1_all_res) / iterations:.4f}")

    # 保存日志
    os.makedirs('../logs', exist_ok=True)
    with open(f'../logs/log_{datasetname}_{args.model_name}.iter{iterations}', 'a+') as f:
        f.write('-------------Original-------------\n')
        f.write(f'All Acc.s:{test_accs}\n')
        f.write(f'All Prec.s:{prec_all}\n')
        f.write(f'All Rec.s:{rec_all}\n')
        f.write(f'All F1.s:{f1_all}\n')
        f.write(f'Average acc.: {sum(test_accs) / iterations}\n')
        f.write(
            f'Average Prec / Rec / F1 (macro): {sum(prec_all) / iterations}, {sum(rec_all) / iterations}, {sum(f1_all) / iterations}\n')

        f.write('\n-------------Adversarial------------\n')
        f.write(f'All Acc.s:{test_accs_res}\n')
        f.write(f'All Prec.s:{prec_all_res}\n')
        f.write(f'All Rec.s:{rec_all_res}\n')
        f.write(f'All F1.s:{f1_all_res}\n')
        f.write(f'Average acc.: {sum(test_accs_res) / iterations}\n')
        f.write(
            f'Average Prec / Rec / F1 (macro): {sum(prec_all_res) / iterations}, {sum(rec_all_res) / iterations}, {sum(f1_all_res) / iterations}\n')