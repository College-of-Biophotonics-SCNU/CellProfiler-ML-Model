import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot(filename, output_path):
    # 加载数据
    df = pd.read_csv(filename)

    # 准备绘图
    plt.figure(figsize=(12, 8))

    # 绘制 boxplot
    for idx, column in enumerate(['Nucleus Compactness',
                                  'Nucleus Mean Intensity', 'Nucleus Area Size',
                                  'Nucleus Eccentricity', 'Nucleus Texture Variance']):
        plt.subplot(2, 3, idx + 1)

        # 绘制箱形图
        sns.boxplot(x='Treatment', y=column, data=df, showfliers=True)  # 显示异常值

        # 添加散点
        sns.stripplot(x='Treatment', y=column, data=df, color=".3", size=3, jitter=True)

        # 获取当前的Axes实例
        ax = plt.gca()
        # 隐藏顶部和右侧的边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('Category')
        plt.ylabel(column.capitalize())

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    plt.show()


if __name__ == "__main__":
    boxplot("24h_data_cleaned.csv", '../../data/result/Hoechst/24h_Boxplot.jpg')
