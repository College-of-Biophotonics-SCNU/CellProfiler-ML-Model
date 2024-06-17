import pickle

import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    use_model = None
    # 创建UAMP模型
    if method_model == "TSNE":
        use_model = TSNE(n_components=2)
    else:
        use_model = umap.UMAP(n_neighbors=10, min_dist=0.1)
    # 对数值型特征进行拟合和转换
    if filter_rule is None:
        embedding = use_model.fit_transform(X)
    else:
        embedding = use_model.fit_transform(X.filter(filter_rule))
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
    print(f"特征数据项中共有{len(loaded_obj)}种时序, 所有特征并集总数为 {len(set_union)}")
    return set_union


class UMAPModel(DataloaderModel):
    def __init__(self, metadata_file):
        # 初始化 MinMaxScaler
        super().__init__()
        # TODO 这里的归一化操作可能对于UMAP模型无效
        self.Parallel = Parallel()  # 多线程处理类
        self.X = None  # metadata 中所有的特征数据
        self.y = None  # metadata 中的label标签，是一个组合标签为 时间和control组合
        self.hours = None  # 记录多少个时间序列
        self.embedding = None  # 归一化后的特征数据
        self.X_valid_feature = {}  # X的有效特征
        self.data_by_hour_split(metadata_file, self.scaler)

    def screen_with_hour(self, pkl_name='20240515_fret_feature'):
        """
        对于每个不同小时的特征进行互信息操作，筛选合适的特征
        多线程处理特征筛选操作
        :return:
        """
        for hour in self.hours:
            X = self.X.iloc[self.y[self.y['Metadata_hour'] == hour].index]
            y = self.y[self.y['Metadata_hour'] == hour]['Metadata_label']
            print(f"{hour} 小时 X 数据格式 {X.shape} y 数据格式 {y.shape}")
            self.X_valid_feature[hour] = self.Parallel.compute(
                screen_feature, (X, y.astype('int'))
            )
            print("筛选特征X在第{}个小时有效特征为 ".format(hour), len(self.X_valid_feature[hour]))
        with open(pkl_name, 'wb') as f:
            pickle.dump(self.X_valid_feature, f)

    def filter_all_and_draw_UAMP(self, save_img, pkl_name=None, control=False, hours=[], only_control=False,
                                 use_model="UMAP"):
        """
        获取特征的并集 实现所有小时的 UMAP 图像
        :param use_model: 
        :param only_control: 仅是只有control组进行降维分析
        :param hours: 需要分析的时间
        :param save_img: 保存图像的位置
        :param pkl_name: 特征保存的位置
        :param control: 按照不同时间点划分 control 组
        :return:
        """
        filter_rule = None
        if pkl_name is not None:
            filter_rule = filter_with_all_feature(pkl_name)
        # 定义颜色列表 选择一个颜色映射，例如 'tab20' 提供了20种区分度较好的颜色
        # cmap = getattr(plt.cm, 'tab20')
        # 生成12个颜色，因为颜色映射是连续的，我们需要将其离散化
        # colors = [cmap(i) for i in np.linspace(0, 1, 10)]
        colors = ['blue', 'Pink', 'red', 'green', 'Orange']
        plt.figure(figsize=(10, 10))
        # 查看是否需要添加按照时间序列的control组
        all_control = [[], []]
        for key, hour in enumerate(self.hours):
            if hour not in hours:
                continue
            index = self.y[self.y['Metadata_hour'] == hour].index
            y = self.y.iloc[index]['Metadata_label']
            X = self.X.iloc[index]
            embedding = feature_reduce(X, filter_rule, use_model)
            if not control:
                all_control[0].extend(embedding[y == 1, 0])
                all_control[1].extend(embedding[y == 1, 1])
            else:
                plt.scatter(embedding[y == 1, 0], embedding[y == 1, 1], label=f'{hour} hour control',
                            color=colors[-key - 1], alpha=1, s=50, edgecolors='none')
            if not only_control:
                plt.scatter(embedding[y == 0, 0], embedding[y == 0, 1], label=f'{hour} hour drug',
                            color=colors[key], edgecolors='none')
        if not control:
            plt.scatter(all_control[0], all_control[1], label='control', color='red',
                        alpha=0.7, s=50, edgecolors='none')
        plt.legend()
        plt.savefig(save_img)
        plt.show()

    def filter_and_draw_UAMP(self, save_img, pkl_name=None):
        """
        按照每个小时进行特征筛选绘制每个小时的特征图像
        :param save_img:
        :param pkl_name:
        :return:
        """
        if pkl_name is not None:
            with open(pkl_name, 'rb') as f:
                loaded_obj = pickle.load(f)
        print(f"特征数据项中共有{len(loaded_obj)}种时序")
        # 假设我们有一个变量 num_subplots，它表示你想要创建的子图数量
        num_subplots = len(self.hours)  # 可以是2, 6, 9等任何整数
        # 计算子图的行数和列数  使用np.ceil来向上取整，确保足够的行来容纳所有子图
        ncols = int(np.ceil(num_subplots ** 0.5))
        nrows = int(np.ceil(num_subplots / ncols))
        # 检测特征值是否为空
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
                if i * ncols + j >= num_subplots:
                    # 方法1: 清除内容并关闭坐标轴
                    ax.cla()  # 清除内容
                    ax.axis('off')  # 关闭坐标轴
                    continue
                filter_rule = loaded_obj[self.hours[subplot_index]]
                ax.set_title(f'UMAP filter by {self.hours[subplot_index]} hour characteristics')
                ax.set_xlabel('X Axis')
                ax.set_ylabel('Y Axis')
                y = self.y[self.y['Metadata_hour'] == self.hours[subplot_index]]['Metadata_label']
                X = self.X.iloc[self.y[self.y['Metadata_hour'] == self.hours[subplot_index]].index]
                # 进行数据过滤和降维
                embedding = self.Parallel.compute(feature_reduce, (X, filter_rule))
                # 在当前子图上绘制散点图
                # 绘制c=0的散点
                plt.scatter(embedding[y == 1, 0], embedding[y == 1, 1], label='control', color='blue', alpha=0.7, s=50,
                            edgecolors='none')
                plt.scatter(embedding[y == 0, 0], embedding[y == 0, 1], label='drug', color='green', alpha=0.7, s=50,
                            edgecolors='none')
                ax.legend()
                # 增加索引以遍历下一个子图
                subplot_index += 1

        # 调整子图之间的间距
        plt.tight_layout()
        plt.savefig(save_img)
        # 显示图形
        plt.show()


if __name__ == '__main__':
    # filter_with_all_feature('../data/20240515_fret_BF_feature_with_hour.pkl')
    model = UMAPModel([
        # '../data/2024515_BF_FRET_BFSingle_No_Texture.csv',
        '../data/2024515_BF_FRET_BFSingle.csv',
        # '../data/2024515_BF_FRET_DDSingle.csv'
    ])
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
    model.filter_all_and_draw_UAMP("../data/result_BF_control_TSNE.jpg",
                                   control=True, only_control=True, hours=[2, 3, 4, 6], use_model="TSNE")

    # 按照 FRET 图像提取有效特征，进行降维分析
    # model.filter_all_and_draw_UAMP("../data/result_FRET_BF_with_1-6_hour_feature_no_control_UMAP.jpg",
    #                                "../data/20240515_fret_BF_feature_with_hour.pkl",
    #                                control=True, hours=[1, 6])
    model.Parallel.close()
