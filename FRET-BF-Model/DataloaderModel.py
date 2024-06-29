import pandas as pd
from dataloader.read_data import read_hour, read_label
from sklearn.model_selection import train_test_split


class DataloaderModel:
    def __init__(self, labels_name=None):
        self.X = None                       # metadata 中所有的特征数据 metadata 中的label标签，是一个组合标签为 时间和control组合
        self.hours = None                   # 记录多少个时间序列
        self.labels = None                  # 存在几种label
        self.embedding = None               # 归一化后的特征数据
        self.scaler = None                  # 标准化方法
        self.labels_name = labels_name       # 标签名称

    def merge_data(self, metadata_file):
        """
        预先加载程序，加载csv文件元数据
        该方法是横向对接数据
        :param metadata_file: 加载的数据文件地址
        :return:
        """
        # 查看是否为多文件加载
        merged_df = {}
        for i in range(len(metadata_file)):
            new_merged = pd.read_csv(metadata_file[i])
            print("文件 {} 的大小为 {}".format(i, new_merged.shape))
            if i == 0:
                merged_df = new_merged
                continue
            else:
                new_merged = new_merged.filter(regex='^(?!Metadata_)|(?!Granularity_)')
            merged_df = merged_df.merge(new_merged, on=['ImageNumber', 'ObjectNumber'])
        merged_df.drop(['ImageNumber', 'ObjectNumber'], axis=1, inplace=True)
        return merged_df

    def concat_data(self, metadata_files):
        """
        按照行进行数据的拼接操作
        """
        print("加载元数据总共具有{}个时间批次的数据".format(len(metadata_files)))
        concat_df = {}
        for i in range(len(metadata_files)):
            if i == 0:
                concat_df = self.merge_data(metadata_files[i])
            else:
                concat_df.concat(self.merge_data(metadata_files[i]), ignore_index=True)
        # 异常值处理操作 对于存在异常值的数值直接去除
        concat_df.dropna(axis=1, how='all', inplace=True)
        concat_df.dropna(inplace=True)
        concat_df.reset_index(drop=True, inplace=True)
        print("所有数据进行合并后的metadata数据大小为 ", concat_df.shape)
        # 进行数据划分 按照 hours 进行数据的划分
        self.hours = read_hour(concat_df)
        self.labels = read_label(concat_df)
        print("总共具有{}种时序的数据, 分别是{}".format(len(self.hours), self.hours))
        print("总共具有{}种标签的数据, 分别是{}".format(len(self.labels), self.labels))
        return concat_df

    def data_by_hour(self, metadata_file, filter_regex=None):
        """
        根据时间进行数据的划分操作， 不划分X 和 y
        :return:
        """
        merged_df = self.concat_data(metadata_file)
        self.X = merged_df.filter(regex='^(?!Metadata_)|Metadata_hour|Metadata_label')
        # 删除重复的列，只保留第一个出现的列
        self.X = self.X.loc[:, ~self.X.columns.duplicated()]
        if filter_regex is not None:
            self.X = self.X.filter(regex=filter_regex + "|Metadata_hour|Metadata_label")
        print("单细胞特征矩阵大小", self.X.shape)


def split_data(X, y):
    """
    数据集划分函数
    :return:
    """
    # 划分数据为训练集和测试集
    # test_size 参数表示测试集的比例，random_state 用于确保每次划分的结果都是一样的（可选）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("训练集大小: ", X_train.shape, y_train.shape, "  测试集大小: ", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test
