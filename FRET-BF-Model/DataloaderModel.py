import pandas as pd

from dataloader.read_data import read_hour
from sklearn.model_selection import train_test_split


class DataloaderModel:
    def __init__(self):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.X = None  # metadata 中所有的特征数据
        self.y = None  # metadata 中的label标签，是一个组合标签为 时间和control组合
        self.hours = None  # 记录多少个时间序列
        self.embedding = None  # 归一化后的特征数据
        self.scaler = None

    def pre_data(self, metadata_file):
        """
        预先加载程序，加载csv文件元数据
        :param metadata_file: 加载的数据文件地址
        :return:
        """
        # 查看是否为多文件加载
        merged_df = {}
        for i in range(len(metadata_file)):
            new_merged = pd.read_csv(metadata_file[i])
            print(f"文件 {i} 的大小为 {new_merged.shape}")
            if i == 0:
                merged_df = new_merged
                continue
            else:
                new_merged = new_merged.filter(regex='^(?!Metadata_)')
            merged_df = merged_df.merge(new_merged, on=['ImageNumber', 'ObjectNumber'])
        merged_df.drop(
            ['ImageNumber', 'ObjectNumber'],
            axis=1, inplace=True)
        # 异常值处理操作 对于存在异常值的数值直接去除
        merged_df.dropna(axis=1, how='all', inplace=True)
        merged_df.dropna(inplace=True)
        merged_df.reset_index(drop=True, inplace=True)
        print("metadata数据大小为 ", merged_df.shape)
        # 进行数据划分 按照 hours 进行数据的划分
        self.hours = read_hour(merged_df)
        print("总共具有{}种时序的数据, 分别是{}".format(len(self.hours), self.hours))
        return merged_df

    def data_by_hour_split(self, metadata_file, scaler=None):
        """
        按照小时进行划分，分为 X 和 y
        :param metadata_file:
        :param scaler:
        :return:
        """
        merged_df = self.pre_data(metadata_file)
        # 对于数据进行划分划分为 X 与 y 数据，也就是细胞特征与label
        self.y = merged_df[['Metadata_hour', 'Metadata_label']]
        self.X = merged_df.filter(regex='^(?!Metadata_)')

        print("y的特征矩阵类型", self.y.shape)
        # 这里对于数据进行归一化操作，TODO 需要注意的是对于 所有数据归一化还是单独小时的数据进行划分
        numeric_data = self.X.select_dtypes(include=['int64', 'float64'])
        # 对数值型特征进行拟合和转换
        if scaler is not None:
            scaled_data = scaler.fit_transform(numeric_data)
        else:
            scaled_data = numeric_data
        # 将结果转换回DataFrame（如果需要）
        self.X = pd.DataFrame(scaled_data, columns=numeric_data.columns)
        print("X的特征矩阵类型", self.X.shape)

    def data_by_hour(self, metadata_file):
        """
        根据时间进行数据的划分操作， 不划分X 和 y
        :return:
        """
        merged_df = self.pre_data(metadata_file)
        self.X = merged_df.filter(regex='^(?!Metadata_)|Metadata_hour|Metadata_label')
        # 删除重复的列，只保留第一个出现的列
        self.X = self.X.loc[:, ~self.X.columns.duplicated()]
        print("X的特征矩阵类型", self.X.shape)


def split_data(X, y):
    """
    数据集划分函数
    :return:
    """
    # 划分数据为训练集和测试集
    # test_size 参数表示测试集的比例，random_state 用于确保每次划分的结果都是一样的（可选）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("训练集大小: ", X_train.shape, y_train.shape, "   测试集大小: ", X_test.shape,
          y_test.shape)
    return X_train, X_test, y_train, y_test
