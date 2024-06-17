from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from DataloaderModel import DataloaderModel, split_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt


class AnalysisModel(DataloaderModel):
    def __init__(self, metadata_file):
        """
        分析模型模板：主要按照数据进行 小时 和 label 之间的划分进行数据分析操作
        :param metadata_file: 加载的模板文件
        """
        super().__init__()
        self.data_by_hour(metadata_file)

    def classify_by_hour(self, hours=None, scaler="StandardScaler", dataname=""):
        categories = []
        values = []
        # 创建一个新的图形
        plt.figure(figsize=(10, 6))
        # 进行归一化操作
        for hour in hours:
            if hour not in self.hours:
                continue
            # 初始化KNN分类器，设置邻居数为3
            knn = KNeighborsClassifier()
            y = self.X.loc[self.X['Metadata_hour'] == hour, ['Metadata_label']]
            X = self.X.iloc[y.index].drop(['Metadata_hour', 'Metadata_label'], axis=1)
            categories.append('{} hour'.format(hour))
            print("{} 小时的数据有 {}".format(hour, (X.shape, y.shape)))
            X_train, X_test, y_train, y_test = split_data(X, y.to_numpy().reshape(-1))
            # 设置对应的归一化操作，便于KNN进行分析
            if scaler == "StandardScaler":
                self.scaler = StandardScaler()
            elif scaler == "MinMaxScaler":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = None
            # 数据标准化（可选步骤，但通常对于KNN是有益的）
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            # 使用训练数据拟合模型
            knn.fit(X_train_scaled, y_train)
            # 对测试集进行预测
            y_pred = knn.predict(X_test_scaled)

            # 评估模型
            accuracy = accuracy_score(y_test, y_pred)
            values.append(accuracy * 100)
            print(f"Accuracy: {accuracy}")

        # 遍历每个柱子，并添加值作为标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2., height,
                         str('%d' % int(height)) + '%',
                         ha='center', va='bottom')

        # 绘制柱状图
        bars = plt.bar(categories, values)
        # 调用函数为柱子添加标签
        autolabel(bars)
        # 添加标题和轴标签
        plt.title(dataname)
        plt.xlabel('Categories')
        plt.ylabel('Accuracy')
        # 设置y轴的范围为0-100
        plt.ylim(0, 100)
        # 显示图形
        plt.savefig('../data/result_analysis_' + dataname + "_classify_by_label.jpg")


class KNNModel(AnalysisModel):
    """
    主要是采用了 KNN 临近值算法
    按照每个小时的数据进行二分类划分操作
    """
    pass


class MLRModel(AnalysisModel):
    pass


if __name__ == "__main__":
    # 采用 KNN 临近算法进行判断
    knn = KNNModel(
        ['../data/2024515_BF_FRET_BFSingle.csv',
         '../data/2024515_BF_FRET_DDSingle.csv']
    )
    knn.classify_by_hour([2, 3, 4, 6], dataname='BF_FRET_feature')

    # 采用多元线性回归算法进行判断预测
