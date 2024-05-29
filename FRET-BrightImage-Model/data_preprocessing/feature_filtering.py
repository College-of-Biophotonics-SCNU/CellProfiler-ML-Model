import csv
import pickle

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif


def remove_feature_Pearson_correlation(X, y):
    """
    筛选与目标值相关的特征记录为表单数据
    :param y:
    :param X:
    :param df:
    :return:
    """
    # 计算相关系数
    # 计算特征和目标变量之间的相关系数
    correlations = X.corrwith(y)

    # 根据相关系数筛选特征
    selected_features = correlations[abs(correlations) > 0.5].index.tolist()  # 假设选择相关系数绝对值大于0.5的特征
    return selected_features


def remove_feature_KBest(X, y, k=50):
    """
    卡方筛选特征方法
    :param X:
    :param y:
    :param k:
    :return:
    """
    # 使用SelectKBest结合chi2指定要保留的特征数量
    selector = SelectKBest(score_func=chi2, k=k)  # 假设保留5个特征
    X_new = selector.fit_transform(X, y)
    # 打印所选特征
    selected_features = X.columns[selector.get_support(indices=True)].tolist()
    return selected_features


def remove_feature_Mutual_Information(X, y, score=0.1):
    """
    利用互信息方法筛选合适的特征
    :param score:
    :param X:
    :param y:
    :return:
    """
    # 计算特征与目标变量之间的互信息
    mutual_info = mutual_info_classif(X, y)
    # 输出所有特征的互信息得分
    feature_scores = pd.Series(mutual_info, index=X.columns)
    print(feature_scores)
    # 假设我们希望选择得分在前50%的特征
    selected_features = X.columns[mutual_info > score]
    print(selected_features)
    return selected_features


def compare_feature(A_filename, B_filename):
    with open('../data/{}.pkl'.format(A_filename), 'rb') as f:
        loaded_A_feature = pickle.load(f)

    with open('../data/{}.pkl'.format(B_filename), 'rb') as f:
        loaded_B_feature = pickle.load(f)

    print(A_filename, " 所包含的特征数量有", len(loaded_A_feature))
    print(B_filename, " 所包含的特征数量有", len(loaded_B_feature))
    max_feature, min_feature = ('loaded_A_feature', 'loaded_B_feature') \
        if len(loaded_A_feature) > len(loaded_B_feature) else ('loaded_B_feature', 'loaded_A_feature')
    print("最大特征数量的特征文件是 ", max_feature)
    # 计算行名称相等的部分数量
    common_indices = [feature for feature in loaded_B_feature if feature in loaded_A_feature]
    number_of_common_indices = len(common_indices)
    print("公共特征数量是", number_of_common_indices)
    # 找出最长的长度
    max_length = max(len(loaded_A_feature), len(loaded_B_feature))

    # 创建一个CSV文件并准备写入数据
    with open('../data/feature_output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头（如果需要）
        writer.writerow([max_feature, min_feature, 'common_indices'])

        # 遍历最长的长度，并一列一列地写入数据
        for i in range(max_length):
            row = [loaded_A_feature[i] if i < len(loaded_A_feature) else '',
                   loaded_B_feature[i] if i < len(loaded_B_feature) else '',
                   common_indices[i] if i < len(common_indices) else '']
            writer.writerow(row)

