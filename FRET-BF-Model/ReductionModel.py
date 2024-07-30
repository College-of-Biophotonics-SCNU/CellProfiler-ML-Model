import pickle

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import preprocessing.feature_filtering as ff
from multiprocesser.Parallel import Parallel
from DataloaderModel import DataloaderModel


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
        selected_feature = ff.remove_feature_Mutual_Information(X, y.astype('int'), 0.1)
    return selected_feature


def feature_reduce(X, filter_rule, method_model="UMAP"):
    """
    特征过滤，过滤无效的特征
    :return:
    """
    # 创建UAMP模型
    if method_model == "TSNE":
        use_model = TSNE()
    else:
        use_model = umap.UMAP(n_neighbors=30, min_dist=0.1)

    # 对数值型特征进行拟合和转换
    if filter_rule is None:
        embedding = use_model.fit_transform(X)
    else:

        X = X.filter(filter_rule)
        print("处理细胞过滤操作，过滤的特征个数为", len(filter_rule), "，矩阵大小为", X.shape)
        print("对应特征为", filter_rule)
        embedding = use_model.fit_transform(X)
    return embedding


def filter_with_all_feature(pkl_name=None):
    """
    对于所有的特征进行特征筛选的过程
    将所有特征取一个并集
    :param pkl_name:
    :return:
    """
    if pkl_name is not None:
        with open(pkl_name, 'rb') as f:
            loaded_obj = pickle.load(f)
    else:
        return None
    set_union = set()
    for hour, value in loaded_obj.items():
        set_union = set_union.union(set(value))
    print("特征数据项中共有{}种时序, 所有特征并集总数为 {}".format(len(loaded_obj), len(set_union)))
    return set_union


class ReductionModel(DataloaderModel):
    def __init__(self, metadata_file, data_name, labels_name=None, filter_regex=None):
        """
        降维算法
        """
        super().__init__(labels_name)
        # 获取单细胞特征，按照小时进行划分数据
        self.Parallel = Parallel()  # 加载线程池
        self.X_valid_feature = {}  # X的有效特征
        self.data_name = data_name  # 数据名称
        self.data_by_hour(metadata_file, filter_regex)

    def screen_with_hour(self):
        """
        筛选合适的有效特征方法，默认采用互信息特征筛选方法
        对于每个不同小时的特征进行互信息操作，筛选合适的特征
        多线程处理特征筛选操作
        :return:
        """
        for hour in self.hours:
            y = self.X.loc[self.X['Metadata_hour'] == hour, 'Metadata_label']
            X = self.X.iloc[y.index].drop(['Metadata_hour', 'Metadata_label'], axis=1)
            print(f"{hour} 小时 X 数据格式 {X.shape} y 数据格式 {y.shape}")

            self.X_valid_feature[hour] = self.Parallel.compute(
                screen_feature, (X, y.astype('int'))
            )
            print("筛选特征X在第{}个小时有效特征为 ".format(hour), len(self.X_valid_feature[hour]))
        with open("../data/features/" + self.data_name + "_features.pkl", 'wb') as f:
            pickle.dump(self.X_valid_feature, f)

    def filter_and_draw_all(self, hour_labels=None, hours=None, labels=None, method_model="UMAP",
                            all_features=False, filter_regex=None):
        """
        获取特征的并集 实现所有小时的 UMAP 图像
        :param all_features: 是否引用所有的数据
        :param method_model: 使用的降维模型
        :param labels: 进行降维的 labels 选择
        :param hours: 需要分析的时间
        :param hour_labels: 按照不同时间点划分 control 组
        :param filter_regex: 正则化过滤不需要的特征
        :return:
        """

        if all_features is True:
            filter_rule = None
        else:
            filter_rule = filter_with_all_feature("../data/features/" + self.data_name + "_features.pkl")
        # 定义颜色列表 选择一个颜色映射，例如 'tab20' 提供了20种区分度较好的颜色
        # cmap = getattr(plt.cm, 'tab20')
        # 生成12个颜色，因为颜色映射是连续的，我们需要将其离散化
        # colors = [cmap(i) for i in np.linspace(0, 1, 12)]
        colors = ['blue', 'Pink', 'red', 'green', 'Orange']
        plt.figure(figsize=(10, 10))
        # 不输入值默认为 所有labels
        if labels is None:
            labels = self.labels
        # 查看那个组别需要进行时间划分操作
        # 默认为 1 表示是对照组数据
        if hour_labels is None:
            hour_labels = self.labels
        # 颜色下标参数
        color_key = 0
        # 进行降维分析
        X = self.X.drop(['Metadata_label', 'Metadata_hour'], axis=1)
        # 过滤一些不需要的特征信息
        if filter_regex is not None:
            X = X.filter(regex=filter_regex).copy()
        # 进行归一化操作
        if self.scaler is not None:
            scaled_data = self.scaler.fit_transform(X)
            scaled_df = pd.DataFrame(scaled_data, columns=X.columns)
        else:
            scaled_df = X
        print("输入的单细胞特征矩阵大小为", scaled_df.shape)
        embeddings = feature_reduce(scaled_df, filter_rule, method_model)
        print("降维后的数据大小为 ", embeddings.shape)
        for label in labels:
            # 判断是否是指定标签内的数据信息 进行按照时间进行划分
            if label in hour_labels:
                for hour in self.hours:
                    # 判断是否是指定分析时间序列上
                    if hours is not None and hour not in hours:
                        continue
                    # 进行不同时间上的散点绘制
                    embedding = embeddings[(self.X['Metadata_hour'] == hour) & (self.X['Metadata_label'] == label), :]
                    print("{} hour {} label 的矩阵大小为 {}".format(hour, label, embedding.shape))
                    plt.scatter(embedding[:, 0], embedding[:, 1],
                                label='{} hour {}'.format(
                                    hour,
                                    self.labels_name[str(label)] if str(label) in self.labels_name else label
                                ),
                                color=colors[- color_key], alpha=1, s=50, edgecolors='none')
                    color_key += 1
            else:
                print("{} label 的矩阵大小为 {}".format(label, embeddings.shape))
                plt.scatter(embeddings[self.X['Metadata_label'] == label, 0],
                            embeddings[self.X['Metadata_label'] == label, 1],
                            label=self.labels_name[str(label)] if str(label) in self.labels_name else label,
                            color=colors[- color_key], edgecolors='none')
                color_key += 1

        plt.legend()
        plt.savefig("../data/result/result_" + self.data_name +
                    "_model_" + method_model +
                    "_labels" + str(labels) +
                    "_allF_" + str(all_features) +
                    "_hour_control" + str(hour_labels) + ".jpg")
        plt.show()

    def filter_and_draw_hour(self, hours=None):
        """
        按照每个小时进行特征筛选绘制每个小时的特征图像
        :return:
        """
        with open("../data/features/" + self.data_name + "_features.pkl", 'rb') as f:
            loaded_obj = pickle.load(f)
        print("特征数据项中共有{}种时序".format(len(loaded_obj)))
        if hours is None:
            hours = self.hours
        # 假设我们有一个变量 num_subplots，它表示你想要创建的子图数量
        num_subplots = len(hours)  # 可以是2, 6, 9等任何整数
        # 计算子图的行数和列数  使用np.ceil来向上取整，确保足够的行来容纳所有子图
        ncols = int(np.ceil(num_subplots ** 0.5))
        nrows = int(np.ceil(num_subplots / ncols))
        # 检测特征值是否为空
        # 创建一个新的图形窗口和子图对象 根据需要调整figsize
        plt.figure(figsize=(ncols * 5, nrows * 5))
        # 定义颜色列表 选择一个颜色映射，例如 'tab20' 提供了20种区分度较好的颜色
        cmap = getattr(plt.cm, 'tab20')
        # 生成12个颜色，因为颜色映射是连续的，我们需要将其离散化
        colors = [cmap(i) for i in np.linspace(0, 1, 12)]
        # 初始化一个索引，用于遍历子图
        subplot_index = 0
        # 假设我们有一些数据要绘制在每个子图上
        for i in range(nrows):
            for j in range(ncols):
                plt.subplot(nrows, ncols, subplot_index + 1)
                ax = plt.gca()  # 获取当前轴
                # 检查索引是否超出了子图数量
                if i * ncols + j >= num_subplots:
                    # 方法1: 清除内容并关闭坐标轴
                    ax.cla()  # 清除内容
                    ax.axis('off')  # 关闭坐标轴
                    continue
                filter_rule = loaded_obj[hours[subplot_index]]
                ax.set_title('Reduction filter by {} hour characteristics'.format(hours[subplot_index]))
                ax.set_xlabel('X Axis')
                ax.set_ylabel('Y Axis')
                y = self.X.loc[self.X['Metadata_hour'] == hours[subplot_index], 'Metadata_label']
                X = self.X.iloc[y.index].drop(['Metadata_hour', 'Metadata_label'], axis=1)
                # 进行数据过滤和降维
                embedding = self.Parallel.compute(feature_reduce, (X, filter_rule))
                # 在当前子图上绘制散点图
                # 绘制c=0的散点
                color_key = 0
                for label in self.labels:
                    plt.scatter(embedding[y == label, 0], embedding[y == label, 1],
                                label=self.labels_name[str(label)] if str(label) in self.labels_name else label,
                                color=colors[color_key],
                                alpha=0.7,
                                s=50,
                                edgecolors='none')
                    color_key += 1
                ax.legend()
                # 增加索引以遍历下一个子图
                subplot_index += 1

        # 调整子图之间的间距
        plt.tight_layout()
        plt.savefig("../data/result/result_" + self.data_name + "_hour_valid_features.jpg")
        # 显示图形
        # plt.show()


class UMAPModel(ReductionModel):
    def __init__(self, metadata_file, data_name, scaler=None, labels_name=None):
        super().__init__(metadata_file, data_name, labels_name)
        if scaler == "StandardScaler":
            self.scaler = StandardScaler()
        elif scaler == "MinMaxScaler":
            self.scaler = MinMaxScaler()


class TSNEModel(ReductionModel):
    def __init__(self, metadata_file, data_name, scaler="StandardScaler", labels_name=None, filter_regex=None):
        super().__init__(metadata_file, data_name, labels_name, filter_regex)
        if scaler == "StandardScaler":
            self.scaler = StandardScaler()
        elif scaler == "MinMaxScaler":
            self.scaler = MinMaxScaler()

    def filter_and_draw_all(self, hour_labels=None, hours=None, labels=None, method_model="TSNE",
                            all_features=False, filter_regex=None):
        super().filter_and_draw_all(hour_labels=hour_labels, hours=hours, labels=labels,
                                    method_model="TSNE", all_features=all_features, filter_regex=filter_regex)


if __name__ == '__main__':
    # model = TSNEModel([
    #     [
    #         '../data/20240616_FRET_BF_Image_BFSingle.csv',
    #         # '../data/2024616_FRET_BF_Image_DDSingle.csv'
    #     ],
    # ],
    #     data_name="20240616_BF",
    #     labels_name={'0': 'drug', '1': 'control', '3': 'roi'},
    #     filter_regex="^AreaShape_"
    # )

    model = UMAPModel([
        [
            '../data/20240628_FRET_BF_Image_BFSingle_4h.csv',
            # '../data/2024616_FRET_BF_Image_DDSingle.csv'
        ],
    ],
        data_name="20240628_BF_all",
        scaler="StandardScaler",
        labels_name={'0': 'drug', '1': 'control', '3': 'roi'},
    )

    # 筛选合适的特征数据
    # model.screen_with_hour()
    model.filter_and_draw_all(hours=[4], all_features=True)
    # model.filter_and_draw_hour([2, 3, 4, 6])

    # 按照不同小时的FRET图像提取有效特征，进行降维分析
    # model.screen_with_hour('../data/20240515_fret_feature_with_hour.pkl')
    # model.filter_and_draw_UAMP("../data/result_FRET_UMAP.jpg", "../data/20240515_fret_feature_with_hour.pkl")

    # 按照不同小时的 FRET + BF 图像提取有效特征，进行降维分析
    # model.screen_with_hour('../data/20240515_fret_BF_feature.pkl')
    # model.filter_and_draw_UAMP("../data/result_FRET_BF_UMAP.jpg", "../data/20240515_fret_BF_feature_with_hour.pkl")

    # 按照所有 FRET + BF 图像提取有效特征，进行降维分析
    # model.filter_all_and_draw_UAMP("../data/result_FRET_BF_all_feature_no_control_UMAP.jpg",
    #                                "../data/20240515_fret_BF_feature_with_hour.pkl",
    #                                control=True)

    # 按照 BF 图像进行control组所有特征的分析，进行降维分析
    # model.filter_all_and_draw_UAMP("../data/result_BF_control_UMAP.jpg",
    #                                control=True, only_control=True, hours=[2, 3, 4, 6])

    # 按照 BF 图像进行不同时间实验组和control组特征分析，进行降维分析
    # model.filter_all_and_draw_UAMP("../data/result_BF_exp_control_UMAP.jpg", hours=[2, 3, 4, 6])

    # 按照 BF 图像进行不同时间实验组和不同时间control组特征分析，进行降维分析
    # model.filter_all_and_draw_UAMP("../data/result_BF_exp_control_with_hour_UMAP.jpg",
    #                                control=True, only_control=False, hours=[2, 3, 4, 6])
    #
    # 按照 BF 图像进行control组所有特征的分析，进行降维分析, 算法采用TSNE
    # model.filter_all_and_draw_UAMP("../data/result_BF_control_TSNE.jpg",
    #                                control=True, only_control=True, hours=[2, 3, 4, 6], use_model="TSNE")

    # 按照 FRET 图像提取有效特征，进行降维分析
    # model.filter_all_and_draw_UAMP("../data/result_FRET_BF_with_1-6_hour_feature_no_control_UMAP.jpg",
    #                                "../data/20240515_fret_BF_feature_with_hour.pkl",
    #                                control=True, hours=[1, 6])
    model.Parallel.close()
