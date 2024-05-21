# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def get_data(is_label=False):
    # 加载data中的cvs数据
    label = {}
    df_BrightSingleCell = pd.read_csv('../data/PC_BrightSingleCell.csv')
    df_DASingleCell = pd.read_csv('../data/PC_DASingleCell.csv')
    df_DDSingleCell = pd.read_csv('../data/PC_DDSingleCell.csv')
    # 加载三个文件夹的数据进行特征合并
    merged_df = pd.merge(df_BrightSingleCell, df_DASingleCell, on=['ImageNumber', 'ObjectNumber'])
    merged_df = pd.merge(merged_df, df_DDSingleCell, on=['ImageNumber', 'ObjectNumber'])
    if is_label:
        label = merged_df['label']
        merged_df.drop(['label'], axis=1)
    merged_df.drop(['Location_Center_X', 'Location_Center_Y', 'Location_Center_Z',
                    'Number_Object_Number'], axis=1, inplace=True)
    return merged_df, label


class TSNEModel:
    # 标准化特征
    scaler = StandardScaler()
    # 初始化KNN分类器
    knn = KNeighborsClassifier(n_neighbors=5)

    def train(self, X_train, y_train):
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        # 在训练集上训练模型
        self.knn.fit(X_train_scaled, y_train)

    def test(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        # 使用训练好的模型在测试集上进行预测
        y_pred = self.knn.predict(X_test_scaled)
        # 计算预测准确率
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)


if __name__ == '__main__':
    data, y = get_data()
    print('data.shape', data.shape)
    print('label中数字有', len(set(y)), '个不同的数字')
    X = data.drop(['ImageNumber', 'ObjectNumber'], axis=1)
    print('data有', X.shape[0], '个样本')
    print('每个样本', X.shape[1], '维数据')
    # 数据降维操作
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, random_state=0, perplexity=13)
    X = tsne.fit_transform(X)
    fig = plt.figure(figsize=(8, 8))  # 指定图像的宽和高
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('FRET-BF t-SNE', fontsize=14)
    print(X.shape)
    # 将数据分割为训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    plt.scatter(X[:, 0], X[:, 1], cmap=plt.cm.Spectral)
    # 显示图像
    plt.show()
    # # TSNE 的模型训练
    # train(X_train, y_train)
    # # 测试数据集
    # test(X_test, y_test)
