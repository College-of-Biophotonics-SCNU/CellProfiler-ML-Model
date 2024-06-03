import preprocessing.feature_filtering as feature_filtering
import pickle


# 比较两个时间点内特征差异
# feature_filtering.compare_feature("feature_2h", "feature_6h")

# 绘制韦恩图进行特征值之间的比对
def compare_valid_feature(filename):
    with open(filename, 'rb') as f:
        loaded_obj = pickle.load(f)
    feature_filtering.draw_vnn(loaded_obj)


if __name__ == "__main__":
    compare_valid_feature("../data/20240515_feature_with_hour.pkl")
