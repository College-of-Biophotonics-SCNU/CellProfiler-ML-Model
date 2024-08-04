import umap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def standard(df, columns):
    """
    标准化数据
    """
    # 创建一个StandardScaler实例
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)
    return df_scaled


def modeling(filenames, labels):
    """
    加载数据 构建模型
    """
    all_data = pd.DataFrame()
    for filename, label in zip(filenames, labels):
        data = pd.read_csv(filename)
        data['label'] = label
        all_data = pd.concat([all_data, data], ignore_index=True)

    all_data.drop(['ImageNumber', 'ObjectNumber'], axis=1, inplace=True)
    features = all_data.columns.drop(['label'])
    features_labels = all_data['label']

    # 数据标准化
    all_data = standard(all_data, features)

    # 初始化UMAP模型
    reducer = umap.UMAP(random_state=42)  # 设置随机种子以保证结果可复现

    # 拟合并转换数据
    embedding = reducer.fit_transform(all_data[features])

    # 如果需要，可以将降维后的结果附加到原始DataFrame
    all_data['embedding_x'] = embedding[:, 0]
    all_data['embedding_y'] = embedding[:, 1]

    # 颜色数组 分别对应红色、绿色、蓝色、黄色
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']

    # 可视化降维后的数据，按标签着色
    plt.figure(figsize=(10, 8))
    for label in labels:
        indexs = features_labels[features_labels == label].index
        plt.scatter(all_data.iloc[indexs]['embedding_x'],
                    all_data.iloc[indexs]['embedding_y'],
                    c=colors[labels.index(label)],
                    cmap='viridis',
                    label=label,
                    alpha=0.7)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP projection of the Hoechst Phenotypic features at 24h')
    plt.legend(loc='upper left', frameon=False)
    plt.savefig("../data/result/Hoechst/UMAP_Hoechst_24h.jpg", dpi=400)
    plt.show()


if __name__ == '__main__':
    modeling(
        [
            '../data/features/HoechstFeatures/A1331852_24h_nucleus.csv',
            '../data/features/HoechstFeatures/ABT-199_24h_nucleus.csv',
            '../data/features/HoechstFeatures/control_24h_nucleus.csv'
        ],
        [
            'A1331852',
            'ABT-199',
            'control'
        ]
    )
