import csv

import preprocessing.feature_filtering as feature_filtering
import pickle


# 比较两个时间点内特征差异
# feature_filtering.compare_feature("feature_2h", "feature_6h")

# 绘制韦恩图进行特征值之间的比对
def compare_valid_feature_setup(filename, save_filename):
    """
    比较两个文件之间的有效数据的交集
    :param filename: 读取文件
    :param save_filename: 保存文件
    :return:
    """
    with open(filename, 'rb') as f:
        loaded_obj = pickle.load(f)
    feature_filtering.draw_vnn(loaded_obj, save_filename)


def record_valid_feature(filename, save_filename):
    """
    记录有效特征数据，利用 Upset 图进行绘制操作
    :param filename:
    :param save_filename:
    :return:
    """
    with open(filename, 'rb') as f:
        loaded_obj = pickle.load(f)
    # 找出列表中的最长长度
    max_length = max(len(value) for key, value in loaded_obj.items())
    # 使用空值（如''或None，取决于你的需求）填充较短的列表

    # 写入CSV文件
    with open(save_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 写入列标题（如果需要的话）
        writer.writerow([f"{key} hour features" for (key, value) in loaded_obj.items()])

        # 逐行写入数据
        for i in range(max_length):
            row_data = [column[i] if len(column) > i else '' for key, column in loaded_obj.items()]
            writer.writerow(row_data)


if __name__ == "__main__":
    # compare_valid_feature_setup("../data/20240515_fret_BF_feature_with_hour.pkl",
    #                             "../data/result_20240515_BF_fret_feature_setup.jpg")
    compare_valid_feature_setup("../data/20240515_fret_feature_with_hour.pkl",
                                "../data/result/result_20240515_fret_feature_setup.jpg")
    # record_valid_feature("../data/20240515_fret_BF_feature_with_hour.pkl", "../data/20240515_BF_fret_features.csv")

