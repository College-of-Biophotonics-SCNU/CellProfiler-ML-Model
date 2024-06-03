import numpy as np


# 定义函数来识别和处理异常值
def remove_outliers_Z_score(df, threshold=3):
    """
    Z-score 方法处理异常值
    :param df: 输入的表单
    :param threshold: 阈值
    :return:
    """
    z_scores = np.abs((df - df.mean()) / df.std())
    df_cleaned = df[(z_scores < threshold).all(axis=1)]
    return df_cleaned



