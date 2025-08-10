import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
import argparse
import numpy as np
import sys, os

sys.path.append(os.getcwd())
from utils.load_data import *
import warnings
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 创建ArgumentParser对象，用于解析命令行参数
parser = argparse.ArgumentParser()

# 添加数据集名称参数，默认值为'politifact'，类型为字符串
parser.add_argument('--dataset_name', default='news', type=str)
# 添加模型名称参数，默认值为'Pretrained-LM'，类型为字符串
parser.add_argument('--model_name', default='Pretrained-LM', type=str)
# 添加迭代次数参数，默认值为2，类型为整数
parser.add_argument('--iters', default=2, type=int)
# 添加批次大小参数，默认值为1，类型为整数
parser.add_argument('--batch_size', default=1, type=int)
# 添加训练轮数参数，默认值为5，类型为整数
parser.add_argument('--n_epochs', default=5, type=int)

# 解析命令行参数，并存储到args对象中
args = parser.parse_args()

# 设置设备为CUDA，用于指定模型和数据在GPU上运行
device = torch.device("cuda")

# 设置随机种子以确保结果的可重复性
torch.manual_seed(0)
np.random.seed(0)
# 设置CuDNN为确定性模式，以确保在使用GPU时结果的一致性
torch.backends.cudnn.deterministic = True
# 当使用多GPU时，为所有GPU设置随机种子
torch.cuda.manual_seed_all(0)


# 定义用于训练的数据集类，包含原始文本和两种增强文本
class NewsDatasetAug(Dataset):
    def __init__(self, texts, aug_texts1, aug_texts2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len):
        """
        初始化数据集类。

        该类用于处理和存储文本数据及其增强版本，同时包括标签和细粒度（fine-grained）标签。
        它还负责通过tokenizer将文本数据编码为适合模型输入的形式，并限制输入长度不超过最大长度。

        参数:
        - texts (list of str): 原始文本数据列表。
        - aug_texts1 (list of str): 经过某种增强方法处理后的文本数据列表1。
        - aug_texts2 (list of str): 经过另一种增强方法处理后的文本数据列表2。
        - labels (list): 文本数据对应的标签列表。
        - fg_label (list): 文本数据对应的细粒度标签列表。
        - aug_fg1 (list): 增强文本数据1对应的细粒度标签列表。
        - aug_fg2 (list): 增强文本数据2对应的细粒度标签列表。
        - tokenizer: 用于对文本数据进行编码的tokenizer对象。
        - max_len (int): 输入文本的最大长度限制。
        """
        # 存储原始文本数据
        self.texts = texts
        # 存储第一种增强方法处理后的文本数据
        self.aug_texts1 = aug_texts1
        # 存储第二种增强方法处理后的文本数据
        self.aug_texts2 = aug_texts2
        # 存储用于文本编码的tokenizer对象
        self.tokenizer = tokenizer
        # 存储文本编码的最大长度限制
        self.max_len = max_len
        # 存储文本数据对应的标签
        self.labels = labels
        # 存储文本数据对应的细粒度标签
        self.fg_label = fg_label
        # 存储第一种增强文本数据对应的细粒度标签
        self.aug_fg1 = aug_fg1
        # 存储第二种增强文本数据对应的细粒度标签
        self.aug_fg2 = aug_fg2

    def __getitem__(self, item):
        """
        根据索引item返回处理后的数据样本。

        该方法主要用于数据集类中，用于获取经过编码和增强处理的文本样本及其标签。

        参数:
        - item: 索引位置，用于获取对应位置的文本和标签数据。

        返回:
        - 一个字典，包含原始文本和两个增强文本的编码表示、对应的注意力掩码、标签以及细粒度标签。
        """
        # 对文本进行编码
        text = self.texts[item]
        aug_text1 = self.aug_texts1[item]
        aug_text2 = self.aug_texts2[item]
        label = self.labels[item]
        fg_label = self.fg_label[item]
        aug_fg1 = self.aug_fg1[item]
        aug_fg2 = self.aug_fg2[item]

        # 使用tokenizer对原始文本进行编码
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                                              padding='max_length', truncation=True, return_token_type_ids=False,
                                              return_attention_mask=True, return_tensors='pt')

        # 使用tokenizer对第一个增强文本进行编码
        aug1_encoding = self.tokenizer.encode_plus(aug_text1, add_special_tokens=True, max_length=self.max_len,
                                                   padding='max_length', truncation=True, return_token_type_ids=False,
                                                   return_attention_mask=True, return_tensors='pt')

        # 使用tokenizer对第二个增强文本进行编码
        aug2_encoding = self.tokenizer.encode_plus(aug_text2, add_special_tokens=True, max_length=self.max_len,
                                                   padding='max_length', truncation=True, return_token_type_ids=False,
                                                   return_attention_mask=True, return_tensors='pt')

        # 返回包含所有必要信息的字典
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'input_ids_aug1': aug1_encoding['input_ids'].flatten(),
            'input_ids_aug2': aug2_encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'attention_mask_aug1': aug1_encoding['attention_mask'].flatten(),
            'attention_mask_aug2': aug1_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'fg_label': torch.FloatTensor(fg_label),
            'fg_label_aug1': torch.FloatTensor(aug_fg1),
            'fg_label_aug2': torch.FloatTensor(aug_fg2),
        }

    def __len__(self):
        """
        获取当前实例中文本元素的数量。

        此方法重写了内置的 __len__ 方法，使其能够通过 len() 函数调用。
        它返回了一个整数，表示实例中 'texts' 属性包含的文本元素数量。

        参数:
        - self: 方法自动绑定的实例对象。

        返回值:
        - int: 'texts' 属性中的文本元素数量。
        """
        return len(self.texts)


# 定义用于测试的数据集类，仅包含原始文本
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        """
        初始化文本数据处理类。

        该类主要用于对给定的文本数据进行预处理，包括分词、编码等操作，并保存相关标签信息。
        它在自然语言处理（NLP）任务中尤其有用，如文本分类、情感分析等。

        参数:
        texts (list of str): 需要处理的文本列表，每个元素为一个字符串类型的文本。
        labels (list): 与文本数据相对应的标签列表，用于监督学习任务。
        tokenizer: 用于文本分词的工具对象，通常是预训练模型的分词器，如BERT的tokenizer。
        max_len (int): 文本处理后的最大长度，包括填充或截断文本以达到统一长度。
        """
        # 保存初始化参数到实例变量，以便于后续的文本处理使用
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        """
        根据索引item返回处理后的文本数据和标签。

        参数:
        - item: 索引值，用于获取对应的文本和标签。

        返回:
        - 一个字典，包含原始文本、编码后的输入ID、注意力掩码和标签。
        """
        # 对文本进行编码
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                                              padding='max_length', truncation=True, return_token_type_ids=False,
                                              return_attention_mask=True, return_tensors='pt')

        # 返回包含文本、编码信息和标签的字典
        return {
            'news_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        """
        获取当前实例中文本元素的数量。

        此方法重写了内置的 __len__ 方法，使其能够通过 len() 函数调用。
        它返回了一个整数，表示实例中 'texts' 属性包含的文本元素数量。

        参数:
        - self: 方法自动绑定的实例对象。

        返回值:
        - int: 'texts' 属性中的文本元素数量。
        """
        return len(self.texts)


# 定义基于RoBERTa的分类模型
class RobertaClassifier(nn.Module):
    """
    RobertaClassifier是一个使用RoBERTa预训练模型的分类器。
    它旨在用于文本分类任务，通过在RoBERTa模型的基础上添加 dropout 和全连接层来实现。

    参数:
    - n_classes (int): 表示分类任务中的类别数量。
    """

    def __init__(self, n_classes):
        """
        初始化函数，用于设置模型的结构。

        参数:
        - n_classes (int): 分类任务的类别数量。
        """
        super(RobertaClassifier, self).__init__()  # 初始化父类

        # 加载预训练的RoBERTa模型
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        # 添加dropout层，防止过拟合
        self.dropout = nn.Dropout(p=0.5)

        # 添加全连接层，输出维度与类别数量相同
        self.fc_out = nn.Linear(self.roberta.config.hidden_size, n_classes)

        # 添加一个用于二进制转换的全连接层
        self.binary_transform = nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        """
        前向传播函数，用于处理输入数据并生成模型输出。

        参数:
        input_ids (Tensor): 输入文本的编码ID，用于标识文本中的每个词或子词。
        attention_mask (Tensor): 注意力掩码，用于区分有效编码和填充编码。

        返回:
        output (Tensor): 主任务的输出，用于多类分类。
        binary_output (Tensor): 二分类任务的输出，用于判断特定条件。
        """
        # 获取RoBERTa模型的输出，并进行分类
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # 提取RoBERTa模型的池化输出
        pooled_outputs = outputs[1]
        # 对池化输出应用dropout，防止过拟合
        pooled_outputs = self.dropout(pooled_outputs)
        # 通过全连接层生成最终的分类输出
        output = self.fc_out(pooled_outputs)
        # 通过二分类转换层生成二分类输出
        binary_output = self.binary_transform(pooled_outputs)

        return output, binary_output


# 创建训练数据加载器
def create_train_loader(contents, contents_aug1, contents_aug2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len,
                        batch_size):
    """
    创建用于训练的DataLoader。

    该函数初始化一个包含原始文本及其两种增强版本的新闻数据集，并为这些数据应用分词和编码。
    然后，它返回一个DataLoader对象，该对象可用于在训练过程中批量迭代数据集。

    参数:
    - contents (list of str): 原始文本内容列表。
    - contents_aug1 (list of str): 第一种增强方法处理后的文本内容列表。
    - contents_aug2 (list of str): 第二种增强方法处理后的文本内容列表。
    - labels (list): 文本的标签列表，用于分类任务。
    - fg_label (int): 前景标签，表示特定类别的标签值。
    - aug_fg1 (list of str): 第一种增强方法处理后的前景文本内容列表。
    - aug_fg2 (list of str): 第二种增强方法处理后的前景文本内容列表。
    - tokenizer: 用于文本分词的Tokenizer对象。
    - max_len (int): 文本编码后的最大长度。
    - batch_size (int): DataLoader在迭代时返回的每个批次的大小。

    返回:
    DataLoader: 用于训练的DataLoader对象，它包装了初始化的新闻数据集。
    """
    # 初始化新闻数据集，包含原始文本和两种增强文本，以及相应的标签
    ds = NewsDatasetAug(texts=contents, aug_texts1=contents_aug1, aug_texts2=contents_aug2, labels=np.array(labels), \
                        fg_label=fg_label, aug_fg1=aug_fg1, aug_fg2=aug_fg2, tokenizer=tokenizer, max_len=max_len)

    # 返回DataLoader对象，用于批量迭代数据集，打乱数据顺序，并使用多线程加载数据
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)


# 创建评估数据加载器
def create_eval_loader(contents, labels, tokenizer, max_len, batch_size):
    """
    创建用于评估的DataLoader对象。

    该函数用于将给定的文本内容和标签转换为适用于模型评估的DataLoader对象。
    它利用一个特定的tokenizer将文本内容tokenize，并与标签一起封装进一个自定义的NewsDataset对象。
    随后，该NewsDataset对象被用来生成一个DataLoader对象，便于在评估过程中批量处理数据。

    参数:
        contents (list of str): 文本内容列表，每个元素是一条文本数据。
        labels (list): 与文本内容对应的标签列表。
        tokenizer: 用于tokenize文本的tokenizer对象。
        max_len (int): 文本tokenize后的最大长度。
        batch_size (int): DataLoader生成的每个batch的大小。

    返回:
        DataLoader: 用于评估的DataLoader对象，提供对文本数据的批量访问。
    """
    # 创建NewsDataset实例，用于封装文本内容、标签、tokenizer以及最大长度参数
    ds = NewsDataset(texts=contents, labels=np.array(labels), tokenizer=tokenizer, max_len=max_len)

    # 返回DataLoader对象，用于生成评估所需的批量数据
    return DataLoader(ds, batch_size=batch_size, num_workers=0)


# 设置随机种子
def set_seed(seed):
    """
    设置随机数种子以确保实验的可重复性。

    参数:
    seed (int): 随机数种子。

    返回:
    无返回值。

    说明:
    本函数通过设置PyTorch和NumPy的随机数种子，确保实验结果具有一致性和可复现性。
    这在深度学习和机器学习的实验中尤为重要，因为它可以帮助调试和验证不同配置对模型性能的影响。
    """
    # 设置PyTorch的随机数种子
    torch.manual_seed(seed)

    # 设置NumPy的随机数种子
    np.random.seed(seed)

    # 设置CuDNN的确定性模式，以确保在使用GPU时结果的一致性
    torch.backends.cudnn.deterministic = True

    # 设置所有GPU的随机数种子
    torch.cuda.manual_seed_all(seed)



# 训练模型
def train_model(tokenizer, max_len, n_epochs, batch_size, datasetname, iter):
    # 加载数据集，包括训练集和测试集，以及测试集的不同表示形式
    x_train, x_test, x_test_res, y_train, y_test = load_articles(datasetname)

    # 创建测试集的数据加载器，用于模型评估
    # 参数包括特征数据、标签数据、分词器、最大序列长度和批次大小
    test_loader = create_eval_loader(x_test, y_test, tokenizer, max_len, batch_size)

    # 创建测试集（不同表示形式）的数据加载器，用于特定情况下的模型评估
    test_loader_res = create_eval_loader(x_test_res, y_test, tokenizer, max_len, batch_size)

    # 初始化模型和优化器
    # 实例化一个Roberta分类器模型，分类数为4，将模型移动到指定设备上
    model = RobertaClassifier(n_classes=4).to(device)
    # 初始化训练损失列表，用于记录每个epoch的训练损失
    train_losses = []
    # 初始化训练准确度列表，用于记录每个epoch的训练准确度
    train_accs = []
    # 使用AdamW优化器初始化模型参数，学习率设置为2e-5
    optimizer = AdamW(model.parameters(), lr=2e-5)
    # 设置总训练步数为10000步
    total_steps = 10000
    # 初始化学习率调度器，使用线性衰减策略，无预热步骤，总训练步数为10000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 训练循环
    for epoch in range(n_epochs):
        # 设置模型为训练模式
        model.train()
        # 加载训练数据，这里的数据经过重新框架处理，以适应模型训练的需要
        x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t = load_reframing(args.dataset_name)
        # 创建训练数据加载器，它将数据集分割成批次，以便在训练过程中进行迭代
        train_loader = create_train_loader(x_train, x_train_res1, x_train_res2, y_train, y_train_fg, y_train_fg_m,
                                           y_train_fg_t, tokenizer, max_len, batch_size)

        # 初始化平均损失列表，用于存储每个批次的损失值
        avg_loss = []
        # 初始化平均准确率列表，用于存储每个批次的准确率
        avg_acc = []
        # 初始化批次索引计数器
        batch_idx = 0

        # 使用tqdm包装训练数据加载器，以显示训练进度
        for Batch_data in tqdm(train_loader):
            # 前向传播
            # 将输入数据和标签移动到指定设备
            input_ids = Batch_data["input_ids"].to(device)
            attention_mask = Batch_data["attention_mask"].to(device)
            input_ids_aug1 = Batch_data["input_ids_aug1"].to(device)
            attention_mask_aug1 = Batch_data["attention_mask_aug1"].to(device)
            input_ids_aug2 = Batch_data["input_ids_aug2"].to(device)
            attention_mask_aug2 = Batch_data["attention_mask_aug2"].to(device)
            targets = Batch_data["labels"].to(device)
            fg_labels = Batch_data["fg_label"].to(device)
            fg_labels_aug1 = Batch_data["fg_label_aug1"].to(device)
            fg_labels_aug2 = Batch_data["fg_label_aug2"].to(device)

            # 对原始和增强的数据进行前向传播
            out_labels, out_labels_bi = model(input_ids=input_ids, attention_mask=attention_mask)
            out_labels_aug1, out_labels_bi_aug1 = model(input_ids=input_ids_aug1, attention_mask=attention_mask_aug1)
            out_labels_aug2, out_labels_bi_aug2 = model(input_ids=input_ids_aug2, attention_mask=attention_mask_aug2)

            # 计算细粒度标签的损失
            fg_criterion = nn.BCELoss()
            # 细粒度损失是模型在原始和两种增强数据上的输出标签的平均损失
            finegrain_loss = (fg_criterion(F.sigmoid(out_labels), fg_labels) + fg_criterion(F.sigmoid(out_labels_aug1),
                                                                                                        fg_labels_aug1) + \
                                          fg_criterion(F.sigmoid(out_labels_aug2), fg_labels_aug2)) / 3

            # 计算原始和增强数据的softmax概率
            out_probs = F.softmax(out_labels_bi, dim=-1)
            aug_log_prob1 = F.log_softmax(out_labels_bi_aug1, dim=-1)
            aug_log_prob2 = F.log_softmax(out_labels_bi_aug2, dim=-1)

            # 计算监督学习损失
            # 使用交叉熵损失函数来衡量模型输出标签与实际目标标签之间的差异
            sup_criterion = nn.CrossEntropyLoss()
            sup_loss = sup_criterion(out_labels_bi, targets)

            # 计算一致性损失
            cons_criterion = nn.KLDivLoss(reduction='batchmean')
            # 一致性损失计算方式：KL散度损失函数，用于衡量两个概率分布之间的差异
            # 这里的一致性损失是两个增强视图的概率分布与原始输出概率分布的KL散度的平均
            cons_loss = 0.5 * cons_criterion(aug_log_prob1, out_probs) + 0.5 * cons_criterion(aug_log_prob2, out_probs)

            # 总损失为监督损失、一致性损失和细粒度损失的加权和
            loss = sup_loss + cons_loss + finegrain_loss

            # 反向传播和优化
            optimizer.zero_grad()  # 清零梯度，避免累加
            loss.backward()  # 损失反向传播，计算模型参数的梯度
            avg_loss.append(loss.item())  # 记录当前损失值，用于后续分析或日志记录
            optimizer.step()  # 更新模型参数，根据梯度进行优化
            scheduler.step()  # 学习率调度，根据训练进度调整学习率

            # 计算训练准确度
            _, pred = out_labels_bi.max(dim=-1)
            # 解释：找出模型预测的最大值所在的位置（即预测的类别），out_labels_bi为模型输出的批次标签的二进制形式

            correct = pred.eq(targets).sum().item()
            # 解释：比较预测类别(pred)与实际类别(targets)，计算预测正确的数量。eq函数用于逐元素比较两个张量是否相等，返回一个包含0和1的张量，1表示对应位置元素相等，0表示不相等。sum()计算所有元素之和，item()将结果转换为Python数值

            train_acc = correct / len(targets)
            # 解释：计算当前批次的训练准确度，correct为预测正确的数量，len(targets)为当前批次的总样本数

            avg_acc.append(train_acc)
            # 解释：将当前批次的训练准确度添加到avg_acc列表中，以便后续计算平均准确度

            batch_idx = batch_idx + 1
            # 解释：批次索引加1，表示处理下一个批次

        # 将平均损失添加到训练损失列表中
        train_losses.append(np.mean(avg_loss))
        # 将平均准确率添加到训练准确率列表中
        train_accs.append(np.mean(avg_acc))

        # 打印当前迭代次数、纪元次数和训练准确率
        print("Iter {:03d} | Epoch {:05d} | Train Acc. {:.4f}".format(iter, epoch, train_acc))

        # 如果当前轮次是训练的最后一轮
        if epoch == n_epochs - 1:
            # 将模型设置为评估模式，以禁用dropout等仅在训练时需要的特性
            model.eval()

            # 初始化列表，用于存储预测的结果
            y_pred = []
            # 初始化列表，用于存储预测结果的原始版本（未经过softmax等处理）
            y_pred_res = []
            # 初始化列表，用于存储真实的标签值
            y_test = []

            # 遍历测试数据加载器，带有进度条显示
            for Batch_data in tqdm(test_loader):
                # 在不计算梯度的情况下进行前向传播，用于测试或验证阶段
                with torch.no_grad():
                    # 将输入数据、注意力掩码和标签转移到指定设备（GPU或CPU）
                    input_ids = Batch_data["input_ids"].to(device)
                    attention_mask = Batch_data["attention_mask"].to(device)
                    targets = Batch_data["labels"].to(device)

                    # 通过模型进行前向传播，获取验证输出
                    _, val_out = model(input_ids=input_ids, attention_mask=attention_mask)

                    # 从输出中获取预测值，即最大概率的索引
                    _, val_pred = val_out.max(dim=1)

                    # 将预测结果和真实标签分别添加到列表中
                    y_pred.append(val_pred)
                    y_test.append(targets)

            # 遍历测试数据加载器，带有进度条显示
            for Batch_data in tqdm(test_loader_res):
                # 在没有梯度计算的上下文中进行操作，以减少内存消耗
                with torch.no_grad():
                    # 将输入ID转移到指定设备
                    input_ids_aug = Batch_data["input_ids"].to(device)
                    # 将注意力掩码转移到指定设备
                    attention_mask_aug = Batch_data["attention_mask"].to(device)
                    # 通过模型传递输入数据，获取模型输出
                    _, val_out_aug = model(input_ids=input_ids_aug, attention_mask=attention_mask_aug)
                    # 从模型输出中找到最大值的索引，作为预测类别
                    _, val_pred_aug = val_out_aug.max(dim=1)
                    # 将预测结果添加到结果列表中
                    y_pred_res.append(val_pred_aug)

            # 将所有预测的输出张量y_pred沿着维度0进行拼接
            y_pred = torch.cat(y_pred, dim=0)
            # 将所有测试的标签张量y_test沿着维度0进行拼接
            y_test = torch.cat(y_test, dim=0)
            # 将所有预测结果的张量y_pred_res沿着维度0进行拼接
            y_pred_res = torch.cat(y_pred_res, dim=0)

            # 计算模型的准确率
            acc = accuracy_score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            # 计算模型的精确度、召回率和F1分数，使用宏平均以考虑所有类别的表现
            precision, recall, fscore, _ = score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy(),
                                                 average='macro')

            # 计算另一个模型或同一模型不同情况下的准确率
            acc_res = accuracy_score(y_test.detach().cpu().numpy(), y_pred_res.detach().cpu().numpy())
            # 同样方法计算该情况下的精确度、召回率和F1分数
            precision_res, recall_res, fscore_res, _ = score(y_test.detach().cpu().numpy(),
                                                             y_pred_res.detach().cpu().numpy(), average='macro')

    # 保存模型的参数状态字典
    # 结合数据集名称和迭代次数生成模型参数文件的路径和文件名
    torch.save(model.state_dict(), '../checkpoints/' + datasetname + '_iter' + str(iter) + '.m')

    # 打印当前迭代轮次的结束标志
    print("-----------------End of Iter {:03d}-----------------".format(iter))

    # 打印当前迭代轮次的全局测试指标，包括准确率、精确度、召回率和F1分数
    print(['Global Test Accuracy:{:.4f}'.format(acc),
           'Precision:{:.4f}'.format(precision),
           'Recall:{:.4f}'.format(recall),
           'F1:{:.4f}'.format(fscore)])

    # 打印指标重置标志
    print("-----------------Restyle-----------------")

    # 打印重置后的全局测试指标，包括准确率、精确度、召回率和F1分数
    print(['Global Test Accuracy:{:.4f}'.format(acc_res),
           'Precision:{:.4f}'.format(precision_res),
           'Recall:{:.4f}'.format(recall_res),
           'F1:{:.4f}'.format(fscore_res)])

    return acc, precision, recall, fscore, acc_res, precision_res, recall_res, fscore_res


if __name__ == '__main__':
    # 从命令行参数中读取数据集名称，用于指定训练所用的数据集
    datasetname = args.dataset_name
    # 从命令行参数中读取批次大小，即每次训练所用的样本数量
    batch_size = args.batch_size
    # 设置输入文本的最大序列长度为512，这是 RoBERTa 模型通常支持的最大长度
    max_len = 512
    # 初始化 RoBERTa 的分词器（Tokenizer），使用的是预训练的 "roberta-base" 模型
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # 从命令行参数中读取训练的总轮次（epoch）数量
    n_epochs = args.n_epochs
    # 从命令行参数中读取训练的总迭代次数（可能用于多次实验或循环训练）
    iterations = args.iters

    # 初始化测试准确率列表
    test_accs = []
    # 初始化精确度、召回率和F1分数的列表，用于存储所有测试数据的结果
    prec_all, rec_all, f1_all = [], [], []
    # 初始化经过调整后的测试准确率列表
    test_accs_res = []
    # 初始化经过调整后的精确度、召回率和F1分数的列表，用于存储所有测试数据的结果
    prec_all_res, rec_all_res, f1_all_res = [], [], []

    # 开始迭代训练模型
    for iter in range(iterations):
        # 设置随机种子以确保结果的可重复性
        set_seed(iter)

        # 训练模型并获取性能指标
        acc, prec, recall, f1, \
        acc_res, prec_res, recall_res, f1_res = train_model(tokenizer,
                                                            max_len,
                                                            n_epochs,
                                                            batch_size,
                                                            datasetname,
                                                            iter)

        # 将本次迭代的测试准确性添加到列表中
        test_accs.append(acc)
        # 将本次迭代的精确率添加到列表中
        prec_all.append(prec)
        # 将本次迭代的召回率添加到列表中
        rec_all.append(recall)
        # 将本次迭代的F1分数添加到列表中
        f1_all.append(f1)
        # 将本次迭代的测试准确性（可能是不同阈值或方法的结果）添加到列表中
        test_accs_res.append(acc_res)
        # 将本次迭代的精确率（可能是不同阈值或方法的结果）添加到列表中
        prec_all_res.append(prec_res)
        # 将本次迭代的召回率（可能是不同阈值或方法的结果）添加到列表中
        rec_all_res.append(recall_res)
        # 将本次迭代的F1分数（可能是不同阈值或方法的结果）添加到列表中
        f1_all_res.append(f1_res)

    print("Total_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
        sum(test_accs) / iterations, sum(prec_all) / iterations, sum(rec_all) / iterations, sum(f1_all) / iterations))

    print("Restyle_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
        sum(test_accs_res) / iterations, sum(prec_all_res) / iterations, sum(rec_all_res) / iterations,
        sum(f1_all_res) / iterations))

    # 打开日志文件以附加模式写入，文件名为包含数据集名称、模型名称和迭代次数的日志文件
    with open('../logs/log_' + datasetname + '_' + args.model_name + '.' + 'iter' + str(iterations), 'a+') as f:
        # 写入原始测试结果的分割线和各项指标
        f.write('-------------Original-------------\n')
        f.write('All Acc.s:{}\n'.format(test_accs))
        f.write('All Prec.s:{}\n'.format(prec_all))
        f.write('All Rec.s:{}\n'.format(rec_all))
        f.write('All F1.s:{}\n'.format(f1_all))
        # 计算并写入准确率、精确度、召回率和F1值的平均值
        f.write('Average acc.: {} \n'.format(sum(test_accs) / iterations))
        f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all) / iterations,
                                                                        sum(rec_all) / iterations,
                                                                        sum(f1_all) / iterations))

        # 写入对抗测试结果的分割线和各项指标
        f.write('\n-------------Adversarial------------\n')
        f.write('All Acc.s:{}\n'.format(test_accs_res))
        f.write('All Prec.s:{}\n'.format(prec_all_res))
        f.write('All Rec.s:{}\n'.format(rec_all_res))
        f.write('All F1.s:{}\n'.format(f1_all_res))
        # 计算并写入对抗测试中准确率、精确度、召回率和F1值的平均值
        f.write('Average acc.: {} \n'.format(sum(test_accs_res) / iterations))
        f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all_res) / iterations,
                                                                        sum(rec_all_res) / iterations,
                                                                        sum(f1_all_res) / iterations))

