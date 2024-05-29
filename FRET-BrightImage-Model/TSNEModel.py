# coding='utf-8'
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class TSNEModel:
    # 标准化特征
    scaler = StandardScaler()
    # 初始化KNN分类器
    knn = KNeighborsClassifier(n_neighbors=5)
    # TSNE算法降维
    tsne = TSNE(n_components=2, random_state=0)

    def __init__(self, root):
        self.X = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.merged_df = None
        self.label = None
        self.root = root

    def pre_data_FRET(self):
        self.merged_df = pd.read_csv(self.root + '/2024515FRET_DDSingle.csv')
        self.merged_df = self.merged_df.drop(['ImageNumber', 'ObjectNumber', 'Metadata_Cell'], axis=1)
        self.merged_df.fillna(0, inplace=True)
        # 异常值处理操作
        self.label = self.merged_df['Metadata_label']
        # self.merged_df = remove_outliers_Z_score(self.merged_df)
        self.X = self.merged_df.filter(regex='^(?!Metadata_)')
        print('label中数字有', len(set(self.label)), '个不同的数字')
        print('X有', self.X.shape[0], '个样本')
        print('每个样本', self.X.shape[1], '维数据')

    def pre_data_BF_and_FRET(self, is_label=False):
        """
        预处理数据操作，将三个分割对象的单细胞特征融合
        :param is_label: 是否具有结果标签
        :return: None
        """
        # 加载data中的cvs数据
        df_BrightSingleCell = pd.read_csv(self.root + '/PC_BrightSingleCell.csv')
        df_DASingleCell = pd.read_csv(self.root + '/PC_DASingleCell.csv')
        df_DDSingleCell = pd.read_csv(self.root + '/PC_DDSingleCell.csv')
        # 加载三个文件夹的数据进行特征合并
        self.merged_df = pd.merge(df_BrightSingleCell, df_DASingleCell, on=['ImageNumber', 'ObjectNumber'])
        self.merged_df = pd.merge(self.merged_df, df_DDSingleCell, on=['ImageNumber', 'ObjectNumber'])
        if is_label:
            self.label = self.merged_df['Metadata_control']
            self.merged_df.drop(['Metadata_control'], axis=1, inplace=True)
        self.merged_df.drop(['Location_Center_X', 'Location_Center_Y', 'Location_Center_Z',
                             'Number_Object_Number'], axis=1, inplace=True)
        self.merged_df = self.merged_df.filter(regex='^(?!Metadata_)')
        self.merged_df.drop_duplicates(inplace=True)
        self.X = self.merged_df.drop(['ImageNumber', 'ObjectNumber'], axis=1)
        self.X.fillna(0, inplace=True)
        print('data.shape', self.X)
        print('label中数字有', len(set(self.label)), '个不同的数字')
        print('data有', self.X.shape[0], '个样本')
        print('每个样本', self.X.shape[1], '维数据')

    def dimension_reduction(self):
        # 数据降维操作
        print('Computing t-SNE embedding')
        self.X = self.scaler.fit_transform(self.X)
        self.X = self.tsne.fit_transform(self.X)
        # 将数据分割为训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.label, test_size=0.2,
                                                                                random_state=42)

    def dimension_reduction_scatter(self):
        fig = plt.figure(figsize=(8, 6))  # 指定图像的宽和高
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('FRET-BF t-SNE', fontsize=14)
        if self.label is not None:
            plt.scatter(self.X[:, 0], self.X[:, 1], c=self.label, cmap=plt.cm.Spectral)
        else:
            plt.scatter(self.X[:, 0], self.X[:, 1], cmap=plt.cm.Spectral)
        # 保存图像
        plt.savefig('../data/result_tsne.jpg')
        # 显示图像
        plt.show()

    def train(self, X_train, y_train):
        # 在训练集上训练模型
        self.knn.fit(X_train, y_train)

    def test(self, X_test, y_test):
        # 使用训练好的模型在测试集上进行预测
        y_pred = self.knn.predict(X_test)
        # 计算预测准确率
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)


if __name__ == '__main__':
    model = TSNEModel('C:/Users/22806/Downloads/result')
    model.pre_data_FRET()
    model.dimension_reduction()
    model.dimension_reduction_scatter()
    # # TSNE 的模型训练
    # train(X_train, y_train)
    # # 测试数据集
    # test(X_test, y_test)
