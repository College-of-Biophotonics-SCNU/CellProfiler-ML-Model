

def read_hour(metadata):
    # 返回数据存在时间列表
    hours = metadata['Metadata_hour'].unique()
    return hours


def read_label(metadata):
    # 返回数据存在的labels列表
    labels = metadata['Metadata_label'].unique()
    return labels
