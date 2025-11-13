import pickle
import numpy as np


def load_webpages(obj):
    """
    加载网页钓鱼检测数据集，包括训练集、标准测试集和对抗性测试集。

    参数:
    - obj: 数据集名称标识符（字符串），用于指定加载哪个数据集

    返回:
    - x_train: 训练集网页文本列表
    - x_test: 标准测试集网页文本列表
    - x_test_res: 对抗性测试集网页文本列表（用于风格迁移后的测试）
    - y_train: 训练集对应的标签列表（0=正常，1=钓鱼）
    - y_test: 测试集对应的标签列表
    - url_train: 训练集URL列表
    - url_test: 测试集URL列表
    """

    # 打印当前加载的数据集名称
    print('Dataset: ', obj)
    # 提示正在加载网页内容
    print("loading web pages")

    # 从指定路径加载训练集数据（使用 pickle 保存的 pkl 文件）
    train_dict = pickle.load(open('../data/web_articles/' + obj + '_train.pkl', 'rb'))
    # 从指定路径加载测试集数据
    test_dict = pickle.load(open('../data/web_articles/' + obj + '_test.pkl', 'rb'))

    # 从指定路径加载对抗性测试集（风格迁移后的测试数据）
    restyle_dict = pickle.load(open('../data/adversarial_test/' + obj + '_test_adv_A.pkl', 'rb'))
    # 注：可根据需要切换到其他对抗测试集文件，如 '_test_adv_B.pkl'、'_test_adv_C.pkl' 等

    # 提取训练集和测试集的网页文本、URL和标签
    x_train, url_train, y_train = train_dict['html'], train_dict['url'], train_dict['labels']  # 网页文本、URL和对应标签
    # print(x_train, url_train, y_train)
    x_test, url_test, y_test = test_dict['html'], test_dict['url'], test_dict['labels']
    # print(x_test, url_test, y_test)

    # 提取对抗性测试集的网页文本
    x_test_res = restyle_dict['html']

    # 返回训练集、测试集、对抗测试集以及对应的URL和标签
    return x_train, x_test, x_test_res, y_train, y_test, url_train, url_test


def load_reframing(obj):
    """
    加载网页增强数据集并进行数据处理。

    该函数根据提供的对象名加载预先处理好的网页数据集，包括不同风格的网页内容和相应的标签。
    它还通过随机选择一半的数据用另一种风格的网页内容进行替换，以增加数据的多样性。

    参数:
    obj (str): 数据集对象的名称，用于构建数据路径。

    返回:
    tuple: 返回四个处理后的数据集和相应的标签。
    """
    print("loading web page augmentations")
    print('Dataset: ', obj)

    # 加载训练数据集，包括标准、欺骗性、中性和紧急警报风格的网页内容
    restyle_dict_train1_1 = pickle.load(open('../data/reframings/' + obj + '_train_objective.pkl', 'rb'))
    restyle_dict_train1_2 = pickle.load(open('../data/reframings/' + obj + '_train_neutral.pkl', 'rb'))
    restyle_dict_train2_1 = pickle.load(open('../data/reframings/' + obj + '_train_emotionally_triggering.pkl', 'rb'))
    restyle_dict_train2_2 = pickle.load(open('../data/reframings/' + obj + '_train_sensational.pkl', 'rb'))

    # 加载细粒度标签数据集，包括原始、标准和欺骗性标签
    finegrain_dict1 = pickle.load(
        open('../data/veracity_attributions/' + obj + '_fake_standards_objective_emotionally_triggering.pkl', 'rb'))
    finegrain_dict2 = pickle.load(
        open('../data/veracity_attributions/' + obj + '_fake_standards_neutral_sensational.pkl', 'rb'))

    # 提取重写后的网页内容并转换为numpy数组
    x_train_res1 = np.array(restyle_dict_train1_1['rewritten'])
    x_train_res1_2 = np.array(restyle_dict_train1_2['rewritten'])
    x_train_res2 = np.array(restyle_dict_train2_1['rewritten'])
    x_train_res2_2 = np.array(restyle_dict_train2_2['rewritten'])

    # 提取细粒度标签
    y_train_fg, y_train_fg_m, y_train_fg_t = finegrain_dict1['orig_fg'], finegrain_dict1['standard_fg'], finegrain_dict1['deceptive_fg']
    y_train_fg2, y_train_fg_m2, y_train_fg_t2 = finegrain_dict2['orig_fg'], finegrain_dict2['standard_fg'], finegrain_dict2['deceptive_fg']

    # 随机选择一半的数据索引
    replace_idx = np.random.choice(len(x_train_res1), len(x_train_res1) // 2, replace=False)

    # 使用另一种风格的网页内容和标签替换随机选择的数据
    x_train_res1[replace_idx] = x_train_res1_2[replace_idx]
    x_train_res2[replace_idx] = x_train_res2_2[replace_idx]
    y_train_fg[replace_idx] = y_train_fg2[replace_idx]
    y_train_fg_m[replace_idx] = y_train_fg_m2[replace_idx]
    y_train_fg_t[replace_idx] = y_train_fg_t2[replace_idx]

    # 返回处理后的数据集和标签
    return x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t
