import pandas as pd


def z_score(df):
    # 计算每列的均值和标准偏差
    mean = df.mean()
    std_dev = df.std()
    # 设定阈值（例如，均值±3倍标准偏差）
    threshold = 3 * std_dev
    # 创建一个布尔掩码来标识哪些行应该保留
    mask = (df - mean).abs() < threshold
    print(df[mask])
    return mask


def read_treatment(filename, output_path):
    # 加载数据
    df = pd.read_csv(filename)
    # 从DataFrame中提取不需要处理的列
    column_to_keep = df[['Treatment']]
    # 删除不需要处理的列
    df_for_z_score = df.drop(columns=['Treatment'])
    mask = z_score(df_for_z_score)
    output_df = df[mask]
    output_df['Treatment'] = column_to_keep
    output_df = output_df.dropna()
    output_df.to_csv(output_path)

if __name__ == "__main__":
    read_treatment("../draw_designs/24h_object_data.csv", '../draw_designs/24_data_cleaned.csv')
