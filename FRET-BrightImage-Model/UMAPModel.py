import pickle

import umap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import data_preprocessing.feature_filtering as ff


class UMAPModel:
    def __init__(self):
        self.merged_df = None
        # 初始化StandardScaler
        self.scaler = StandardScaler()
        self.reducer = umap.UMAP()
        self.X = None
        self.y = None
        self.embedding = None

    def pre_data_FRET(self):
        self.merged_df = pd.read_csv('../data/2024515_2h_FRET_DATA.csv')
        self.merged_df = self.merged_df.drop(['ImageNumber', 'ObjectNumber', 'Metadata_Cell'], axis=1)
        self.merged_df.fillna(0, inplace=True)
        # 异常值处理操作
        self.y = self.merged_df['Metadata_label']
        self.X = self.merged_df.filter(regex='^(?!Metadata_)')
        print('X的数据格式', self.X.shape)
        print('y的数据格式', self.y.shape)

    def feature_filter(self, is_screen_feature=False, pkl_name='simple_feature'):
        print("筛选合适的特征！！！")
        if is_screen_feature:
            with open('../data/{}.pkl'.format(pkl_name), 'rb') as f:
                loaded_obj = pickle.load(f)
            self.X = self.X.filter(loaded_obj)
        # 分离出数值型特征
        numeric_data = self.X.select_dtypes(include=['int64', 'float64'])
        # 初始化StandardScaler
        # 对数值型特征进行拟合和转换
        scaled_data = self.scaler.fit_transform(numeric_data)
        # 将结果转换回DataFrame（如果需要）
        self.X = pd.DataFrame(scaled_data, columns=numeric_data.columns)
        print('X的数据格式', self.X.shape)

    def screen_feature(self, method="Mutual_Information", pkl_name='simple_feature'):
        selected_feature = self.X.columns
        if method == "Mutual_Information":
            selected_feature = ff.remove_feature_Mutual_Information(self.X, self.y, 0.1)
        with open('../data/{}.pkl'.format(pkl_name), 'wb') as f:
            pickle.dump(selected_feature, f)
        return selected_feature

    def reduce(self):
        # 对数值型特征进行拟合和转换
        self.embedding = self.reducer.fit_transform(self.X)

    def draw(self):
        # 可视化降维结果
        plt.scatter(self.embedding[:, 0], self.embedding[:, 1], c=self.y, cmap='viridis')
        plt.title('UMAP projection of 2h FRET image dataset using 6h feature', fontsize=15)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        # 存储处理图像
        plt.savefig('../data/result_FRET_2h_UMAP_Use_6h_feature.jpg')
        plt.show()


if __name__ == '__main__':
    model = UMAPModel()
    # 图像加载
    model.pre_data_FRET()
    # 图像的特征筛选，默认采用互信息筛选法
    # model.screen_feature(pkl_name="feature_2h")
    # 图像特征过滤操作，过滤筛选得出的特征，并将数据进行归一化操作
    model.feature_filter(True, pkl_name="feature_6h")
    # 图像特征降维操作
    model.reduce()
    # 输出降维结果
    model.draw()

