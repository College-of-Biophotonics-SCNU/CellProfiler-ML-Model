import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['Category1', 'Category2', 'Category3', 'Category4', 'Category5']
values = [10, 15, 7, 19, 13]

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制柱状图
plt.bar(categories, values)

# 添加标题和轴标签
plt.title('My Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')

# 显示图形
plt.show()