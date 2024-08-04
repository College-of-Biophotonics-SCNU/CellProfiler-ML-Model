import numpy as np
from sklearn.datasets import make_blobs
from umap import UMAP
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# 生成模拟数据
data, labels = make_blobs(n_samples=300, centers=4, n_features=10, random_state=42)

# 使用UMAP进行降维
umap = UMAP(n_components=2, random_state=42)
umap_data = umap.fit_transform(data)

def plot_umap_with_elliptical_contours(umap_data, labels, title):
    # 创建绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 定义颜色映射
    cmap = plt.cm.get_cmap('tab10')

    # 创建网格用于计算密度
    delta = 0.025
    x = np.arange(np.min(umap_data[:, 0]), np.max(umap_data[:, 0]), delta)
    y = np.arange(np.min(umap_data[:, 1]), np.max(umap_data[:, 1]), delta)
    X, Y = np.meshgrid(x, y)

    # 对每个簇绘制散点图和轮廓线
    for i, label in enumerate(np.unique(labels)):
        # 提取簇中的数据
        cluster_data = umap_data[labels == label]

        # 计算核密度估计
        k = gaussian_kde(cluster_data.T)
        Z = k(np.vstack([X.ravel(), Y.ravel()]))
        Z = Z.reshape(X.shape)

        # 绘制轮廓线
        # levels 参数定义了轮廓线的等高线水平，这里选择一些特定的水平来产生椭圆状轮廓
        ax.contour(X, Y, Z, colors='k', linewidths=2, levels=[0.1])

        # 填充轮廓线内的区域
        ax.contourf(X, Y, Z, colors=cmap(i), alpha=0.2, levels=[0.1])

    # 绘制散点图
    for i, label in enumerate(np.unique(labels)):
        cluster_data = umap_data[labels == label]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], c=cmap(i), label=f"Cluster {label}", s=50, edgecolors='k', linewidth=1.5, alpha=0.7)

    # 添加图例
    ax.legend(title='Clusters', loc='upper right')

    # 设置标题
    ax.set_title(title)

    # 显示图表
    plt.show()

# 绘制UMAP散点图
plot_umap_with_elliptical_contours(umap_data, labels, 'UMAP Clustering with Elliptical Contours and Colors')