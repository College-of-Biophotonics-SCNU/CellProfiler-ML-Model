import pickle

import numpy as np
import umap
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing.feature_filtering as ff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataloader.read_data import read_hour
from multiprocesser.Parallel import Parallel


def screen_feature(X, y, method="Mutual_Information"):
    """
    特征筛选流程, 默认采用互信息方法
    :param X:
    :param y:
    :param method:
    :return:
    """
    selected_feature = X.columns
    if method == "Mutual_Information":
        selected_feature = ff.remove_feature_Mutual_Information(X, y, 0.1)
    return selected_feature


def feature_reduce(X, filter_rule):
    """
    特征过滤，过滤无效的特征
    :return:
    """
    # 创建UAMP模型
    umap_model = umap.UMAP()
    # 对数值型特征进行拟合和转换
    embedding = umap_model.fit_transform(X.filter(filter_rule))
    return embedding


class UMAPModel:
    def __init__(self, metadata_file):
        # 初始化 MinMaxScaler
        self.scaler = MinMaxScaler()
        self.Parallel = Parallel()      # 多线程处理类
        self.X = None                   # metadata 中所有的特征数据
        self.y = None                   # metadata 中的label标签，是一个组合标签为 时间和control组合
        self.hours = None               # 记录多少个时间序列
        self.embedding = None           # 归一化后的特征数据
        self.X_valid_feature = {}       # X的有效特征
        self.pre_data(metadata_file)

    def pre_data(self, metadata_file):
        """
        预先加载程序，加载csv文件元数据
        :param metadata_file:
        :return:
        """
        merged_df = pd.read_csv(metadata_file)
        merged_df.drop(
            ['ImageNumber', 'ObjectNumber', 'Metadata_Cell', 'Metadata_FileLocation', 'Metadata_channel'],
            axis=1, inplace=True)
        # 异常值处理操作 对于存在异常值的数值直接去除
        merged_df.dropna(inplace=True)
        merged_df.reset_index(drop=True, inplace=True)
        print("加载的metadata数据大小为 ", merged_df.shape)
        # 进行数据划分 按照 hours 进行数据的划分
        self.hours = read_hour(merged_df)
        print("总共具有{}种时序的数据".format(len(self.hours)))
        # 对于数据进行划分划分为 X 与 y 数据，也就是细胞特征与label
        self.y = merged_df[['Metadata_hour', 'Metadata_label']]
        self.X = merged_df.filter(regex='^(?!Metadata_)')
        print("y的特征矩阵类型", self.y.shape)
        # 这里对于数据进行归一化操作，TODO 需要注意的是对于 所有数据归一化还是单独小时的数据进行划分
        numeric_data = self.X.select_dtypes(include=['int64', 'float64'])
        # 对数值型特征进行拟合和转换
        scaled_data = self.scaler.fit_transform(numeric_data)
        # 将结果转换回DataFrame（如果需要）
        self.X = pd.DataFrame(scaled_data, columns=numeric_data.columns)
        print("X的特征矩阵类型", self.X.shape)

    def screen_with_hour(self, pkl_name='20240515_feature'):
        """
        对于每个不同小时的特征进行互信息操作，筛选合适的特征
        多线程处理特征筛选操作
        :return:
        """
        pkl_name += "_with_hour"
        for hour in self.hours:
            self.X_valid_feature[hour] = self.Parallel.compute(
                screen_feature,
                (self.X.iloc[self.y[self.y['Metadata_hour'] == hour].index],
                 self.y[self.y['Metadata_hour'] == hour]['Metadata_label'])
            )
            print("筛选特征X在第{}个小时有效特征为 ".format(hour), len(self.X_valid_feature[hour]))
        with open('../data/{}.pkl'.format(pkl_name), 'wb') as f:
            pickle.dump(self.X_valid_feature, f)

    def screen_with_all(self, pkl_name='20240515_feature'):
        """
        对于所有的特征进行特征筛选的过程
        :param pkl_name:
        :return:
        """
        # TODO 融入多种类型样本计算计算
        pkl_name += "_with_all"
        pass

    def filter_and_draw_UAMP(self,save_img, pkl_name=None):
        if pkl_name is not None:
            with open(pkl_name, 'rb') as f:
                loaded_obj = pickle.load(f)

        # 假设我们有一个变量 num_subplots，它表示你想要创建的子图数量
        num_subplots = len(self.hours)  # 可以是2, 6, 9等任何整数
        # 计算子图的行数和列数  使用np.ceil来向上取整，确保足够的行来容纳所有子图
        ncols = int(np.ceil(num_subplots ** 0.5))
        nrows = int(np.ceil(num_subplots / ncols))

        # 创建一个新的图形窗口和子图对象 根据需要调整figsize
        plt.figure(figsize=(ncols * 5, nrows * 5))
        # 初始化一个索引，用于遍历子图
        subplot_index = 0
        # 假设我们有一些数据要绘制在每个子图上
        for i in range(nrows):
            for j in range(ncols):
                plt.subplot(nrows, ncols, subplot_index + 1)
                ax = plt.gca()  # 获取当前轴
                # 检查索引是否超出了子图数量
                if i * ncols + j >= num_subplots + 1:
                    # 方法1: 清除内容并关闭坐标轴
                    ax.cla()  # 清除内容
                    ax.axis('off')  # 关闭坐标轴
                    continue
                filter_rule = loaded_obj[self.hours[subplot_index]]
                y = self.y[self.y['Metadata_hour'] == self.hours[subplot_index]]['Metadata_label']
                X = self.X.iloc[self.y[self.y['Metadata_hour'] == self.hours[subplot_index]].index]
                # 进行数据过滤和降维
                embedding = self.Parallel.compute(feature_reduce, (X, filter_rule))
                # 在当前子图上绘制散点图
                plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
                ax.set_title(f'UMAP filter by {self.hours[subplot_index]} hour characteristics')
                ax.set_xlabel('X Axis')
                ax.set_ylabel('Y Axis')

                # 增加索引以遍历下一个子图
                subplot_index += 1

        # 调整子图之间的间距
        plt.tight_layout()
        plt.savefig(save_img)
        # 显示图形
        plt.show()


if __name__ == '__main__':
    model = UMAPModel('../data/2024515_FRET_DATA.csv')
    # 图像的特征筛选，默认采用互信息筛选法
    # model.screen_with_hour()
    # # 图像特征过滤操作，过滤筛选得出的特征，并将数据进行归一化操作
    model.filter_and_draw_UAMP("../data/result_FRET_UMAP.jpg", "../data/20240515_feature_with_hour.pkl")
    model.Parallel.close()
    # # 图像特征降维操作
    # model.reduce()
    # # 输出降维结果
    # model.draw()
